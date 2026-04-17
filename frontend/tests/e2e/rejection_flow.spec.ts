/**
 * E2E: 退件 → 編輯欄位 → 補件流程（mock API）
 */
import { test, expect, type Page } from '@playwright/test'

const BASE = process.env.PLAYWRIGHT_BASE_URL ?? 'http://localhost:3950'

async function injectToken(page: Page) {
  await page.addInitScript(() => {
    localStorage.setItem('modelhub_access_token', 'e2e-test-token')
    localStorage.setItem('modelhub_userinfo', JSON.stringify({
      sub: 'test-user',
      preferred_username: 'e2e-tester',
    }))
  })
}

const REJECTED_SUB = {
  id: 2,
  req_no: 'MH-2026-010',
  req_name: '退件測試模型',
  product: 'AICAD',
  company: 'HurricaneEdge',
  submitter: 'e2e',
  purpose: '測試',
  priority: 'P2',
  model_type: 'detection',
  status: 'rejected',
  rejection_reasons: JSON.stringify(['資料量不足', '缺少驗證集']),
  rejection_note: '請補充至少 3000 張訓練圖',
  resubmit_count: 0,
  class_list: 'cat,dog',
  dataset_source: 'Kaggle',
  dataset_count: '500 張',
  dataset_val_count: null,
  dataset_test_count: null,
  kaggle_dataset_url: '',
  dataset_status: 'ready',
  created_at: new Date().toISOString(),
}

test.describe('退件補件流程', () => {
  test('展開欄位編輯、修改、補件 resubmit', async ({ page }) => {
    // mock GET /api/submissions/MH-2026-010
    let subStatus = 'rejected'
    await page.route('**/api/submissions/MH-2026-010', async (route) => {
      if (route.request().method() === 'GET') {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({ ...REJECTED_SUB, status: subStatus }),
        })
      } else if (route.request().method() === 'PATCH') {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({ ...REJECTED_SUB, dataset_count: '3500 張' }),
        })
      }
    })

    // mock registry
    await page.route('**/api/registry/by-req/MH-2026-010', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify([]),
      })
    })

    // mock resubmit
    await page.route('**/api/submissions/MH-2026-010/resubmit', async (route) => {
      subStatus = 'submitted'
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ req_no: 'MH-2026-010', action: 'resubmit', status: 'submitted' }),
      })
    })

    // mock submissions list
    await page.route('**/api/submissions/', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify([REJECTED_SUB]),
      })
    })

    await injectToken(page)
    await page.goto(`${BASE}/submissions/MH-2026-010`)

    // 應顯示退件 banner
    await expect(page.locator('text=退件缺失項')).toBeVisible({ timeout: 5000 })
    await expect(page.locator('text=資料量不足')).toBeVisible()

    // 展開欄位編輯
    await page.click('button:has-text("編輯需求單欄位")')
    await expect(page.locator('text=修改欄位後點')).toBeVisible()

    // 修改資料集大小
    await page.fill('input[value="500 張"]', '3500 張')

    // 儲存變更
    await page.click('button:has-text("儲存變更")')

    // 填寫補件說明並送審
    await page.fill('textarea[placeholder="請說明補件後修改了什麼..."]', '已補充資料集至 3500 張，並加入驗證集 700 張')
    await page.click('button:has-text("補件 resubmit")')

    // 應重新 query（狀態變成 submitted）
    await expect(page.locator('text=補件 resubmit')).not.toBeVisible({ timeout: 5000 })
  })
})
