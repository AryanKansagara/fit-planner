import { expect, test as base, type Page, type Route } from '@playwright/test'

/** Hosts the page asks for on its own (webfonts). They are blocked, not stubbed: the app must work without them. */
const TOLERATED_BLOCKED = new Set(['fonts.googleapis.com', 'fonts.gstatic.com'])

export interface Net {
  /** Hosts other than the preview server that the page tried to reach and that no test stubbed. */
  blocked: string[]
  /** Uncaught exceptions in the page. */
  pageErrors: string[]
}

/**
 * Every test gets a signed-out browser with a fresh IndexedDB, all non-local traffic aborted
 * (nothing reaches Hugging Face, Google Fonts or Supabase), and a check at the end that the page threw
 * nothing uncaught and asked for no external host that the test did not stub on purpose.
 */
export const test = base.extend<{ net: Net; page: Page }>({
  net: async ({}, use) => { await use({ blocked: [], pageErrors: [] }) },
  page: async ({ page, net, baseURL }, use) => {
    const local = new URL(baseURL!).host
    await page.route(url => new URL(url).host !== local && /^https?:/.test(url.protocol), (route: Route) => {
      net.blocked.push(new URL(route.request().url()).hostname)
      return route.abort('blockedbyclient')
    })
    page.on('pageerror', e => net.pageErrors.push(e.message))
    await use(page)
    expect(net.pageErrors, 'uncaught page errors').toEqual([])
    const unexpected = [...new Set(net.blocked)].filter(h => !TOLERATED_BLOCKED.has(h))
    expect(unexpected, 'external hosts the app tried to reach').toEqual([])
    // Signed out means the Supabase client is never even requested.
    expect(net.blocked.filter(h => h.includes('supabase')), 'supabase requests').toEqual([])
  },
})

export { expect }
