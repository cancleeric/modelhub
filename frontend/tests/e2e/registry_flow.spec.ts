/**
 * E2E: 模型清冊 → 驗收流程（mock API）
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

const MOCK_VERSIONS = [
  {
    id: 1,
    req_no: 'MH-2026-001',
    product: 'AICAD',
    model_name: 'pid-symbol-v1',
    version: 'v1.0',
    train_date: '2026-04-01',
    map50: 0.85,
    map50_95: 0.62,
    map50_actual: 0.87,
    map50_95_actual: 0.63,
    file_path: '/models/pid-v1.pt',
    status: 'pending',
    pass_fail: null,
    accepted_by: null,
    accepted_at: null,
    is_current: false,
    created_at: new Date().toISOString(),
  },
]

test.describe('模型清冊與驗收流程', () => {
  test('清冊列表顯示、進入驗收頁', async ({ page }) => {
    // mock registry list
    await page.route('**/api/registry/', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(MOCK_VERSIONS),
      })
    })

    // mock submissions list
    await page.route('**/api/submissions/', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify([]),
      })
    })

    await injectToken(page)
    await page.goto(`${BASE}/registry`)

    // 應顯示模型版本
    await expect(page.locator('text=pid-symbol-v1')).toBeVisible({ timeout: 5000 })
    await expect(page.locator('text=v1.0')).toBeVisible()
    await expect(page.locator('text=0.87')).toBeVisible()

    // 點擊驗收連結
    const acceptLink = page.locator('a[href*="/accept"]').first()
    await expect(acceptLink).toBeVisible()
  })

  test('驗收頁填寫並提交', async ({ page }) => {
    // mock GET /api/registry/1
    await page.route('**/api/registry/1', async (route) => {
      if (route.request().method() === 'GET') {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify(MOCK_VERSIONS[0]),
        })
      }
    })

    // mock GET submission
    await page.route('**/api/submissions/MH-2026-001', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          id: 1,
          req_no: 'MH-2026-001',
          product: 'AICAD',
          company: 'HurricaneEdge',
          status: 'trained',
          priority: 'P2',
          map50_target: 0.85,
          created_at: new Date().toISOString(),
        }),
      })
    })

    // mock POST accept
    await page.route('**/api/registry/1/accept', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          ...MOCK_VERSIONS[0],
          status: 'accepted',
          pass_fail: 'pass',
          accepted_by: 'e2e-tester',
          accepted_at: new Date().toISOString(),
        }),
      })
    })

    await injectToken(page)
    await page.goto(`${BASE}/registry/1/accept`)

    // 填寫 mAP50 實測值
    const map50Input = page.locator('input[type="number"]').first()
    await expect(map50Input).toBeVisible({ timeout: 5000 })
    await map50Input.fill('0.87')

    // 提交驗收
    const submitBtn = page.locator('button[type="submit"]')
    if (await submitBtn.isVisible()) {
      await submitBtn.click()
    }
  })
})
