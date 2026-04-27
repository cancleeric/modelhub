# ADR: Markdown Rendering — react-markdown + remark-gfm

**Date**: 2026-04-26
**Status**: Accepted
**Deciders**: CTO
**Context**: M22 Discussion System Phase 3 需要在前端 render discussion comment 的 Markdown 內容。

---

## 問題

Discussion 留言的 `body_markdown` 欄位為 Markdown 格式字串，前端需要安全地將其 render 成 HTML，支援：

- `**bold**`, `_italic_`, `code block`
- GFM checklist（`- [ ] item`）用於退件原因顯示
- 程式碼高亮（Phase 3 優化）

## 選項比較

| 方案 | Bundle size | GFM 支援 | XSS 安全 | 維護度 |
|------|------------|----------|----------|--------|
| **react-markdown + remark-gfm** | ~45KB | 是（外掛） | 預設安全 | 高（週 30M+ 下載）|
| marked | ~25KB | 是 | 需自行 sanitize | 高 |
| markdown-it | ~30KB | 透過外掛 | 需自行 sanitize | 高 |
| 自建正規表達式解析 | ~1KB | 否 | 自行控制 | 低 |

## 決策

採用 **react-markdown** (`^9.x`) + **remark-gfm** (`^4.x`)。

理由：
1. **React 原生**：直接接受 `children` string，不需要 `dangerouslySetInnerHTML`，XSS 安全。
2. **GFM checklist**：remark-gfm 支援 `- [ ] item`，退件原因 checklist 直接 render。
3. **元件化**：可自訂 `components` prop 覆蓋各 HTML 元素（code block 加 copy 按鈕等）。
4. **週下載量 30M+**：社群活躍，Issue response 快。

## 使用方式（Phase 3 參考）

```tsx
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

<ReactMarkdown remarkPlugins={[remarkGfm]}>
  {comment.body_markdown}
</ReactMarkdown>
```

## 注意事項

- Phase 2（此 ADR）：只 `npm install` 套件，**不改任何前端 code**（UI Phase 3 再整合）。
- `allowedElements` 或 `disallowedElements` prop 可在 Phase 3 白名單 HTML 標籤（建議限制 `<script>` `<iframe>` 等）。
- 不使用 `rehype-raw`，避免 raw HTML 注入。

## 相關工單

- M22 Phase 3：CommentThread / CommentEditor UI 整合
