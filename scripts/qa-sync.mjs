// QA: exercise the same table calls sync.ts makes, with a throwaway account, then delete it.
import { readFileSync } from 'node:fs'
import { randomUUID } from 'node:crypto'
import { createClient } from '@supabase/supabase-js'
const env = Object.fromEntries(readFileSync(new URL('../.env', import.meta.url), 'utf8').split('\n').filter(l => l.includes('=') && !l.startsWith('#')).map(l => [l.slice(0, l.indexOf('=')), l.slice(l.indexOf('=') + 1).trim()]))
const sb = createClient(env.VITE_SUPABASE_URL, env.VITE_SUPABASE_PUBLISHABLE_KEY, { auth: { persistSession: false } })
const email = `qa-fit-planner-${Date.now()}@example.com`, password = randomUUID()
const ok = (label, cond, extra = '') => { console.log(`${cond ? 'PASS' : 'FAIL'} ${label} ${extra}`); if (!cond) process.exitCode = 1 }
const up = await sb.auth.signUp({ email, password })
ok('sign up returns a session', !!up.data.session, up.error?.message ?? '')
const id = randomUUID(), snap = randomUUID()
const project = { id, name: 'QA plan', model: { kind: 'sample', config: '{}' }, workload: { context: 8192 }, tp: 4, pp: 1 }
let r = await sb.from('projects').upsert({ id, app: 'fit-planner', name: 'QA plan', data: { project }, updated_at: new Date().toISOString() })
ok('upsert project', !r.error, r.error?.message ?? '')
r = await sb.from('project_versions').insert([{ id: snap, project_id: id, label: '4x p150a', data: { id: snap, label: '4x p150a', createdAt: Date.now() } }])
ok('insert snapshot', !r.error, r.error?.message ?? '')
r = await sb.from('projects').select('id,name,data,updated_at').eq('app', 'fit-planner')
ok('read back own project', r.data?.length === 1 && r.data[0].data.project.tp === 4, r.error?.message ?? '')
r = await sb.from('project_versions').select('id').in('project_id', [id])
ok('read back snapshot', r.data?.length === 1)
const anon = createClient(env.VITE_SUPABASE_URL, env.VITE_SUPABASE_PUBLISHABLE_KEY, { auth: { persistSession: false } })
r = await anon.from('projects').select('id').eq('id', id)
ok('anon cannot see the row (RLS)', !r.data?.length)
r = await sb.rpc('delete_my_account')
ok('delete_my_account', !r.error, r.error?.message ?? '')
r = await sb.auth.signInWithPassword({ email, password })
ok('account is gone', !!r.error, email)
