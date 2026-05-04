"""
ModelHub Connector（Phase 3）

向 brain-cloud 注冊 ModelHub 圖像分類模型，橋接推理請求。

連線流程：
1. 連線 brain-cloud /ws/llm-provider
2. 動態查 ModelHub /v1/models → 取得 model list（加 "modelhub/" 前綴）
   查不到時 fallback 到 MODEL_SLUGS env
3. 送 register（帶 provider_api_key、models）
4. 收到 llm_request → 呼叫 LlmHandler → ModelHub /v1/chat/completions
5. 斷線自動重連（指數退避：0s, 5s, 15s, 30s）
"""

import asyncio
import json
import logging
import logging.handlers
import os
import signal
import sys
import uuid

import httpx
import structlog
import websockets

from llm_handler import LlmHandler

# ── Structlog JSON 設定（與 brain-llm-connector 一致）─────────────────────────
_LOG_MAX_BYTES = 100 * 1024 * 1024  # 100 MB
_LOG_BACKUP_COUNT = 7

_log_handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
_connector_log_file = os.environ.get("CONNECTOR_LOG_FILE", "")
if _connector_log_file:
    _log_dir = os.path.dirname(_connector_log_file)
    if _log_dir:
        os.makedirs(_log_dir, exist_ok=True)
    _rh = logging.handlers.RotatingFileHandler(
        _connector_log_file,
        maxBytes=_LOG_MAX_BYTES,
        backupCount=_LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    _log_handlers.append(_rh)

_shared_processors = [
    structlog.contextvars.merge_contextvars,
    structlog.processors.TimeStamper(fmt="iso"),
    structlog.processors.add_log_level,
    structlog.stdlib.add_logger_name,
]

logging.basicConfig(
    handlers=_log_handlers,
    level=logging.INFO,
    format="%(message)s",
)

_formatter = structlog.stdlib.ProcessorFormatter(
    processors=[
        structlog.stdlib.ProcessorFormatter.remove_processors_meta,
        structlog.processors.JSONRenderer(),
    ],
    foreign_pre_chain=_shared_processors,
)
for _h in _log_handlers:
    _h.setFormatter(_formatter)

structlog.configure(
    processors=_shared_processors + [
        structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger("modelhub_connector")


def _new_trace_id() -> str:
    return str(uuid.uuid4())[:8]


# ── 設定（從環境變數讀取）────────────────────────────────────────────────────
BRAIN_CLOUD_WS_URL = os.environ.get(
    "BRAIN_CLOUD_WS_URL",
    "ws://localhost:8932/ws/llm-provider",
)
PROVIDER_API_KEY = os.environ.get("BRAIN_PROVIDER_API_KEY", "")
PROVIDER_ID = os.environ.get("PROVIDER_ID", "modelhub-connector")
PROVIDER_VERSION = "1.0.0"

MODELHUB_API_URL = os.environ.get("MODELHUB_API_URL", "http://localhost:8950")
MODELHUB_API_KEY = os.environ.get("MODELHUB_API_KEY", "")

# fallback model list（當 /v1/models 查不到時使用）
_DEFAULT_SLUGS = "modelhub/mh-2026-010,modelhub/mh-2026-011"
MODEL_SLUGS_FALLBACK: list[str] = [
    m.strip()
    for m in os.environ.get("MODEL_SLUGS", _DEFAULT_SLUGS).split(",")
    if m.strip()
]

# 重連退避策略（與 brain-llm-connector 一致）
_RECONNECT_DELAYS = [0, 5, 15, 30, 60, 120, 300]
_RECONNECT_MAX_DELAY = 300


async def _fetch_models_from_modelhub() -> list[str]:
    """
    動態查 ModelHub /v1/models，回傳 ["modelhub/<req_no>", ...] 格式。
    查不到或出錯時回傳空 list（呼叫方負責 fallback）。
    """
    url = f"{MODELHUB_API_URL}/v1/models"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                url,
                headers={"x-api-key": MODELHUB_API_KEY},
            )
        if resp.status_code == 200:
            data = resp.json()
            models_raw = data.get("models", [])
            slugs = [
                f"modelhub/{m['req_no'].lower()}"
                for m in models_raw
                if m.get("req_no")
            ]
            logger.info(
                "從 ModelHub 取得 model list",
                count=len(slugs),
                models=slugs,
            )
            return slugs
        else:
            logger.warning(
                "ModelHub /v1/models 回傳非 200",
                status=resp.status_code,
                body=resp.text[:200],
            )
            return []
    except Exception as e:
        logger.warning("查詢 ModelHub /v1/models 失敗", error=str(e))
        return []


class ModelHubConnectorDaemon:
    def __init__(self):
        self._running = True
        self._handler = LlmHandler()
        self._reconnect_attempt = 0

    def _next_delay(self) -> int:
        idx = min(self._reconnect_attempt, len(_RECONNECT_DELAYS) - 1)
        return _RECONNECT_DELAYS[idx]

    async def run(self):
        logger.info(
            "ModelHub Connector 啟動",
            url=BRAIN_CLOUD_WS_URL,
            provider_id=PROVIDER_ID,
            modelhub_api_url=MODELHUB_API_URL,
        )

        if not PROVIDER_API_KEY:
            logger.error("BRAIN_PROVIDER_API_KEY 未設定，無法連線")
            sys.exit(1)

        if not MODELHUB_API_KEY:
            logger.warning("MODELHUB_API_KEY 未設定，將使用 MODEL_SLUGS fallback")

        while self._running:
            delay = self._next_delay()
            if delay > 0:
                logger.info(
                    "reconnect",
                    attempt=self._reconnect_attempt,
                    delay=delay,
                    max_delay=_RECONNECT_MAX_DELAY,
                )
                await asyncio.sleep(delay)

            try:
                await self._connect_and_handle()
                if self._reconnect_attempt > 0:
                    logger.info(
                        "重連成功，重置重連計數",
                        previous_attempts=self._reconnect_attempt,
                    )
                self._reconnect_attempt = 0
            except asyncio.CancelledError:
                logger.info("Connector 收到取消信號，退出")
                break
            except Exception as e:
                self._reconnect_attempt += 1
                next_delay = self._next_delay()
                logger.warning(
                    "WS 連線中斷",
                    attempt=self._reconnect_attempt,
                    next_retry_delay=next_delay,
                    error=str(e),
                )

        logger.info("ModelHub Connector 已停止")

    async def _connect_and_handle(self):
        # 動態查詢 model list，查不到就 fallback
        models = await _fetch_models_from_modelhub()
        if not models:
            models = MODEL_SLUGS_FALLBACK
            logger.info(
                "使用 MODEL_SLUGS fallback",
                models=models,
            )

        ssl_ctx = None
        if BRAIN_CLOUD_WS_URL.startswith("wss://"):
            import ssl
            ssl_ctx = ssl.create_default_context()
            ssl_ctx.check_hostname = False
            ssl_ctx.verify_mode = ssl.CERT_NONE

        async with websockets.connect(
            BRAIN_CLOUD_WS_URL,
            ssl=ssl_ctx,
            ping_interval=None,
            open_timeout=30,
        ) as ws:
            logger.info("已連線 brain-cloud", url=BRAIN_CLOUD_WS_URL)

            # ── 送 register ────────────────────────────────────────────────
            await ws.send(json.dumps({
                "type": "register",
                "provider_id": PROVIDER_ID,
                "provider_api_key": PROVIDER_API_KEY,
                "models": models,
                "version": PROVIDER_VERSION,
            }))

            # 等待 registered 確認
            raw = await asyncio.wait_for(ws.recv(), timeout=15.0)
            resp = json.loads(raw)
            if resp.get("type") == "registered":
                logger.info(
                    "Provider 登錄成功",
                    provider_id=resp.get("provider_id"),
                    models=resp.get("models"),
                )
            else:
                raise RuntimeError(f"預期 registered，收到：{resp}")

            # ── 心跳 task ──────────────────────────────────────────────────
            async def _heartbeat():
                while True:
                    await asyncio.sleep(20)
                    try:
                        await ws.send(json.dumps({"type": "pong"}))
                    except Exception:
                        break

            heartbeat_task = asyncio.create_task(_heartbeat())

            try:
                async for raw_msg in ws:
                    try:
                        data = json.loads(raw_msg)
                    except json.JSONDecodeError:
                        logger.warning("收到非合法 JSON，略過")
                        continue

                    msg_type = data.get("type")

                    if msg_type == "ping":
                        await ws.send(json.dumps({"type": "pong"}))

                    elif msg_type == "llm_request":
                        asyncio.create_task(self._handle_llm_request(ws, data))

                    else:
                        logger.debug("未知訊息", msg_type=msg_type)

            finally:
                heartbeat_task.cancel()

    async def _handle_llm_request(self, ws, data: dict):
        """處理 brain-cloud 下發的 llm_request。"""
        request_id = data.get("request_id", "")
        model = data.get("model", "")
        messages = data.get("messages", [])
        stream = data.get("stream", True)
        call_type = data.get("call_type", "chat")
        tenant_id = data.get("tenant_id", "")
        agent_id = data.get("agent_id", "")

        trace_id = data.get("trace_id") or _new_trace_id()

        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            trace_id=trace_id,
            tenant_id=tenant_id,
            request_id=request_id,
        )

        logger.info(
            "llm_request 收到",
            model=model,
            agent_id=agent_id,
            stream=stream,
            call_type=call_type,
        )

        if not request_id or not messages:
            await ws.send(json.dumps({
                "type": "llm_error",
                "request_id": request_id,
                "message": "request_id 或 messages 為空",
            }))
            return

        try:
            if stream:
                await self._handler.handle_stream(
                    ws=ws,
                    request_id=request_id,
                    model=model,
                    messages=messages,
                    call_type=call_type,
                    tenant_id=tenant_id,
                    agent_id=agent_id,
                )
            else:
                await self._handler.handle_non_stream(
                    ws=ws,
                    request_id=request_id,
                    model=model,
                    messages=messages,
                    call_type=call_type,
                    tenant_id=tenant_id,
                    agent_id=agent_id,
                )
        except Exception as e:
            import traceback
            logger.error(
                "llm_request 處理失敗",
                model=model,
                agent_id=agent_id,
                error=str(e),
                traceback=traceback.format_exc(),
            )
            try:
                await ws.send(json.dumps({
                    "type": "llm_error",
                    "request_id": request_id,
                    "message": str(e) or repr(e),
                }))
            except Exception as send_e:
                logger.warning(
                    "送 llm_error 失敗（WS 已斷）",
                    error=str(send_e),
                )
        finally:
            structlog.contextvars.clear_contextvars()


async def _main():
    daemon = ModelHubConnectorDaemon()

    loop = asyncio.get_running_loop()

    def _shutdown(sig):
        logger.info("收到信號，準備退出", signal=sig.name)
        daemon._running = False
        for task in asyncio.all_tasks(loop):
            task.cancel()

    for s in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(s, _shutdown, s)

    await daemon.run()


if __name__ == "__main__":
    asyncio.run(_main())
