/**
 * E2E: 新增草稿 → 儲存 → 送審 → 核准流程（mock API）
 */
import { test, expect, type Page } from '@playwright/test'

const BASE = process.env.PLAYWRIGHT_BASE_URL ?? 'http://localhost:3950'

// 共用 mock token（dev 環境 SKIP_ROLE_CHECK=true 時有效）
async function injectToken(page: Page) {
  await page.addInitScript(() => {
    localStorage.setItem('modelhub_access_token', 'e2e-test-token')
    localStorage.setItem('modelhub_userinfo', JSON.stringify({
      sub: 'test-user',
      preferred_username: 'e2e-tester',
      email: 'e2e@test.local',
    }))
  })
}

test.describe('需求單草稿與送審流程', () => {
  test('建立草稿並送審', async ({ page }) => {
    // mock POST /api/submissions/ — 建立草稿
    await page.route('**/api/submissions/', async (route) => {
      if (route.request().method() === 'POST') {
        await route.fulfill({
          status: 201,
          contentType: 'application/json',
          body: JSON.stringify({
            submission: {
              id: 1,
              req_no: 'MH-2026-099',
              req_name: 'E2E 測試模型',
              product: 'AICAD',
              company: 'HurricaneEdge',
              submitter: 'e2e',
              purpose: '測試用',
              priority: 'P2',
              model_type: 'detection',
              class_list: 'cat,dog',
              status: 'draft',
              created_at: new Date().toISOString(),
              map50_target: 0.9,
            },
            warnings: [],
            suggestions: [],
          }),
        })
      } else {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify([]),
        })
      }
    })

    // mock PATCH /api/submissions/MH-2026-099
    await page.route('**/api/submissions/MH-2026-099', async (route) => {
      if (route.request().method() === 'PATCH') {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({ req_no: 'MH-2026-099', status: 'draft' }),
        })
      }
    })

    // mock POST actions/submit
    await page.route('**/api/submissions/MH-2026-099/actions/submit', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ req_no: 'MH-2026-099', action: 'submit', status: 'submitted' }),
      })
    })

    // mock GET /api/submissions/MH-2026-099
    await page.route('**/api/submissions/MH-2026-099', async (route) => {
      if (route.request().method() === 'GET') {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({
            id: 1,
            req_no: 'MH-2026-099',
            req_name: 'E2E 測試模型',
            product: 'AICAD',
            company: 'HurricaneEdge',
            status: 'submitted',
            priority: 'P2',
            created_at: new Date().toISOString(),
          }),
        })
      }
    })

    await injectToken(page)
    await page.goto(`${BASE}/submit`)

    // 填寫必填欄位
    await page.fill('[name="req_name"]', 'E2E 測試模型')
    await page.fill('[name="company"]', 'HurricaneEdge')
    await page.fill('[name="purpose"]', '這是 E2E 測試用的業務描述')
    await page.fill('[name="map50_target"]', '0.9')
    await page.fill('[name="class_list"]', 'cat\ndog')
    await page.fill('[name="dataset_source"]', 'Kaggle')
    await page.fill('[name="dataset_count"]', '1000 張')
    await page.fill('[name="label_format"]', 'YOLO')
    await page.fill('[name="expected_delivery"]', '2026-06-01')

    // 儲存草稿（不跳轉）
    await page.click('button:has-text("儲存草稿")')
    await expect(page.locator('text=草稿已儲存')).toBeVisible({ timeout: 5000 })

    // 應顯示 req_no
    await expect(page.locator('text=MH-2026-099')).toBeVisible({ timeout: 3000 })

    // 送出審核
    await page.click('button:has-text("送出審核")')

    // 應跳轉到詳情頁
    await expect(page).toHaveURL(/\/submissions\/MH-2026-099/, { timeout: 5000 })
  })
})
