import { defineConfig, devices } from '@playwright/test'

// The suite runs against the production build served by `vite preview`.
// Set E2E_PORT to use a different port (default 4173). The port must be free:
// the server is started with --strictPort and the run fails instead of reusing one.
const port = Number(process.env.E2E_PORT ?? 4173)
if (!Number.isInteger(port) || port < 1 || port > 65535) throw new Error(`E2E_PORT must be a port number, got "${process.env.E2E_PORT}"`)
const origin = `http://127.0.0.1:${port}`

export default defineConfig({
  testDir: './e2e',
  outputDir: './test-results',
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: 0,
  workers: process.env.CI ? 2 : undefined,
  reporter: process.env.CI ? [['list'], ['html', { open: 'never' }]] : [['list']],
  timeout: 30_000,
  expect: { timeout: 7_000 },
  use: {
    baseURL: origin,
    viewport: { width: 1280, height: 900 },
    trace: 'retain-on-failure',
    // Every test starts signed out, with empty storage and a fixed locale/timezone.
    locale: 'en-US',
    timezoneId: 'UTC',
    acceptDownloads: true,
  },
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'], viewport: { width: 1280, height: 900 } } }],
  webServer: {
    command: `node node_modules/vite/bin/vite.js preview --host 127.0.0.1 --port ${port} --strictPort`,
    url: origin,
    // Never attach to a server somebody else started: it may serve a stale or different build.
    reuseExistingServer: false,
    timeout: 60_000,
    stdout: 'ignore',
    stderr: 'pipe',
  },
})
