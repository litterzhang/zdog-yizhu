import assert from 'node:assert/strict'
import { test } from 'node:test'
import { AccountStore, hasVerifiedAccount, type AccountSnapshot, type TokenStorage } from '../src/services/account'
import { AuthApi, AuthError, type AuthTransport } from '../src/services/auth-api'
import { MockJournal } from '../src/services/mock-journal'

const user = { id: 'verified-sub', nickname: '窗边观察者' }
const token = 'test-only-token'
const okUser = { statusCode: 200, data: { code: 200, user } }
const okToken = { statusCode: 200, data: { code: 200, access_token: token } }

test('private journal UI is visible only for an authenticated account', () => {
  const states: AccountSnapshot['status'][] = ['guest', 'loading', 'error', 'authenticated']
  for (const status of states) {
    assert.equal(hasVerifiedAccount({ status, user: null, error: '' }), false)
    assert.equal(hasVerifiedAccount({ status, user, error: '' }), status === 'authenticated')
  }
})

function setup(transport: AuthTransport = async () => okUser, initial: unknown = '') {
  let saved = initial
  const resets: string[] = []
  const storage: TokenStorage = {
    read: () => saved,
    write: (value) => { saved = value },
    remove: () => { saved = '' },
  }
  const api = new AuthApi(transport)
  const store = new AccountStore(api, storage, (reason) => { resets.push(reason) })
  return { api, store, storage, resets, saved: () => saved }
}

function deferred<T>() {
  let resolve!: (value: T) => void
  const promise = new Promise<T>((done) => { resolve = done })
  return { promise, resolve }
}

test('consent gates login before any credential or network operation', async () => {
  const { store } = setup()
  let called = false
  await assert.rejects(store.signIn(false, async () => { called = true; return token }))
  assert.equal(called, false)
  assert.equal(store.getSnapshot().status, 'guest')
})

test('WeChat code exchange then /me verification precede persistence; no raw token in UI state', async () => {
  const calls: Parameters<AuthTransport>[0][] = []
  const { store, api, saved, resets } = setup(async (input) => {
    calls.push(input)
    if (input.path === '/wechat') return okToken
    assert.equal(saved(), '')
    return okUser
  })
  await store.signIn(true, () => api.wechat('temporary-code'))
  assert.deepEqual(calls, [
    { path: '/wechat', method: 'POST', data: { code: 'temporary-code' } },
    { path: '/me', method: 'GET', token },
  ])
  assert.equal(saved(), token)
  assert.deepEqual(store.getSnapshot(), { status: 'authenticated', user, error: '' })
  assert.equal(JSON.stringify(store.getSnapshot()).includes(token), false)
  assert.deepEqual(resets, ['login'])
})

test('H5 sends trimmed username but exact password only to login endpoint', async () => {
  const api = new AuthApi(async (input) => {
    assert.deepEqual(input, { path: '/login', method: 'POST', data: { username: 'gardener', password: ' keep spaces ' } })
    return okToken
  })
  assert.equal(await api.password(' gardener ', ' keep spaces '), token)
  await assert.rejects(api.password(' ', 'x'))
  await assert.rejects(api.wechat(' '))
})

test('credential without a verified server user never counts as login', async () => {
  for (const data of [{ code: 200, user: { id: '' } }, '<html>gateway</html>', { user }]) {
    const { store, saved } = setup(async () => ({ statusCode: 200, data }))
    await assert.rejects(store.signIn(true, async () => token))
    assert.equal(store.getSnapshot().user, null)
    assert.equal(saved(), '')
  }
})

test('invalid tokens and wrong response envelopes are rejected', async () => {
  for (const data of [
    { code: 200 }, { code: 200, access_token: 'bad token' },
    { code: 200, access_token: 'x'.repeat(8193) }, { access_token: token },
  ]) {
    const api = new AuthApi(async () => ({ statusCode: 200, data }))
    await assert.rejects(api.wechat('one-use-code'), AuthError)
  }
})

test('saved session is hidden until /me completes on restore', async () => {
  const pending = deferred<typeof okUser>()
  const { store, saved } = setup(() => pending.promise, token)
  const restoring = store.restore()
  assert.equal(store.getSnapshot().status, 'loading')
  assert.equal(store.getSnapshot().user, null)
  pending.resolve(okUser)
  await restoring
  assert.deepEqual(store.getSnapshot().user, user)
  assert.equal(saved(), token)
})

test('401 clears stored token and private state; no demo account fallback', async () => {
  const { store, saved, resets } = setup(async () => ({ statusCode: 401, data: {} }), token)
  await store.restore()
  assert.equal(saved(), '')
  assert.equal(store.getSnapshot().status, 'guest')
  assert.equal(store.getSnapshot().user, null)
  assert.deepEqual(resets, ['expired'])
})

test('network or 503 failure retains token for explicit retry without claiming login', async () => {
  for (const status of [0, 503]) {
    let failed = true
    const { store, saved, resets } = setup(async () => {
      if (!failed) return okUser
      if (status === 0) throw new Error('sensitive network payload')
      return { statusCode: status, data: {} }
    }, token)
    await store.restore()
    assert.equal(saved(), token)
    assert.equal(store.getSnapshot().status, 'error')
    assert.equal(store.getSnapshot().user, null)
    assert.equal(store.getSnapshot().error.includes('sensitive'), false)
    assert.deepEqual(resets, [])
    failed = false
    await store.restore()
    assert.deepEqual(store.getSnapshot().user, user)
  }
})

test('logout invalidates late /me restore responses', async () => {
  const pending = deferred<typeof okUser>()
  const { store, saved } = setup(() => pending.promise, token)
  const restoring = store.restore()
  store.signOut()
  pending.resolve(okUser)
  await restoring
  assert.equal(saved(), '')
  assert.equal(store.getSnapshot().status, 'guest')
})

test('logout invalidates late WeChat exchange and concurrent submissions are blocked', async () => {
  const pending = deferred<string>()
  const { store, saved } = setup()
  const signingIn = store.signIn(true, () => pending.promise)
  await assert.rejects(store.signIn(true, async () => token), /稍候/)
  store.signOut()
  pending.resolve(token)
  await assert.rejects(signingIn, /取消/)
  assert.equal(saved(), '')
  assert.equal(store.getSnapshot().status, 'guest')
})

test('logout during login identity verification cannot repersist the token', async () => {
  const pending = deferred<typeof okUser>()
  const { store, saved } = setup(() => pending.promise)
  const signingIn = store.signIn(true, async () => token)
  await Promise.resolve()
  store.signOut()
  pending.resolve(okUser)
  await assert.rejects(signingIn, /取消/)
  assert.equal(saved(), '')
})

test('storage failures are explicit, with no authenticated fallback', async () => {
  const { store, storage, saved } = setup()
  storage.write = () => { throw new Error('disk failure') }
  await assert.rejects(store.signIn(true, async () => token), /存储/)
  assert.equal(store.getSnapshot().user, null)
  assert.equal(saved(), '')
  storage.remove = () => { throw new Error('disk failure') }
  assert.throws(() => store.signOut(), /未能清除/)
  assert.equal(store.getSnapshot().status, 'error')
})

test('login/logout clear mock journal without assigning trusted identity to demo records', async () => {
  const journal = new MockJournal()
  const { api, storage } = setup()
  const store = new AccountStore(api, storage, () => journal.clear())
  assert.ok(journal.getSnapshot().plants.length)
  await store.signIn(true, async () => token)
  assert.equal(journal.getSnapshot().plants.length, 0)
  store.signOut()
  assert.equal(journal.getSnapshot().observations.length, 0)
})

test('HTTP errors offer bounded safe messages rather than raw auth-service responses', async () => {
  for (const statusCode of [400, 401, 403, 429, 502, 503, 504]) {
    const api = new AuthApi(async () => ({ statusCode, data: { message: 'private-upstream-details' } }))
    await assert.rejects(api.wechat('code'), (error: unknown) =>
      error instanceof AuthError && error.status === statusCode && !error.message.includes('private-upstream'))
  }
})

test('subscribers are notified and can unsubscribe', async () => {
  const { store } = setup()
  let count = 0
  const unsubscribe = store.subscribe(() => { count++ })
  await store.signIn(true, async () => token)
  assert.equal(count, 2)
  unsubscribe()
  store.signOut()
  assert.equal(count, 2)
})
