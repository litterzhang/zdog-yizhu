import assert from 'node:assert/strict'
import { test } from 'node:test'
import { RemoteJournal, type ServerSnapshot } from '../src/services/remote-journal'
import type { Recognition } from '../src/types/domain'

const snapshot = (owner: string): ServerSnapshot => ({
  species: [], plants: [], observations: [], engine: 'ready',
  recognitions: [{
    id: `record-${owner}`, ownerId: owner, photo: { id: 'photo', origin: 'remote', uri: '' },
    status: 'succeeded', rawResult: null, confirmedSpeciesId: null,
    createdAt: '2026-09-20T10:00:00Z', failure: null,
  }],
})
function deferred<T>() {
  let resolve!: (value: T) => void
  const promise = new Promise<T>((done) => { resolve = done })
  return { promise, resolve }
}

test('remote journal restores server records rather than seeding demo data', async () => {
  const calls: string[] = []
  const store = new RemoteJournal(async (path) => { calls.push(path); return snapshot('alice') }, () => {})
  assert.equal(store.getSnapshot().recognitions.length, 0)
  await store.connect('alice')
  assert.equal(store.getSnapshot().recognitions[0].id, 'record-alice')
  assert.deepEqual(calls, ['/journal'])
  store.clear()
  assert.equal(store.getSnapshot().recognitions.length, 0)
  assert.deepEqual(calls, ['/journal']) // Logout does not delete cloud data.
})

test('late response from another account cannot restore private records', async () => {
  const old = deferred<ServerSnapshot>()
  let first = true
  const store = new RemoteJournal(() => {
    if (first) { first = false; return old.promise }
    return Promise.resolve(snapshot('bob'))
  }, () => {})
  const pending = store.connect('alice')
  await store.connect('bob')
  old.resolve(snapshot('alice'))
  await pending
  assert.equal(store.getSnapshot().recognitions[0].ownerId, 'bob')
  store.clear()
})

test('server ownership mismatch is rejected before rendering', async () => {
  const store = new RemoteJournal(async () => snapshot('bob'), () => {})
  await store.connect('alice')
  assert.equal(store.getSnapshot().recognitions.length, 0)
  assert.match(store.getSnapshot().error!, /身份/)
})

test('server failures are visible and can be retried', async () => {
  let fail = true
  const store = new RemoteJournal(async () => {
    if (fail) throw new Error('网络不可用')
    return snapshot('alice')
  }, () => {})
  await store.connect('alice')
  assert.match(store.getSnapshot().error!, /网络/)
  fail = false
  await store.refresh()
  assert.equal(store.getSnapshot().error, '')
  assert.equal(store.getSnapshot().recognitions.length, 1)
})

test('recognition creation carries stable request key but never a client ownerId', async () => {
  const store = new RemoteJournal(async (path, data) => {
    if (!data) return snapshot('alice')
    assert.equal(path, '/recognitions')
    assert.deepEqual(data, { photoId: 'private-photo', requestId: 'stable-request' })
    return { item: snapshot('alice').recognitions[0], snapshot: snapshot('alice') }
  }, () => {})
  await store.connect('alice')
  assert.equal((await store.startRecognition('private-photo', 'stable-request')).id, 'record-alice')
})

test('late mutation after logout cannot publish or navigate with old data', async () => {
  const pending = deferred<unknown>()
  const store = new RemoteJournal(async (_path, data) => data ? pending.promise : snapshot('alice'), () => {})
  await store.connect('alice')
  const task = store.startRecognition('photo', 'request')
  store.clear()
  pending.resolve({ item: snapshot('alice').recognitions[0], snapshot: snapshot('alice') })
  await assert.rejects(task, /账号/)
  assert.equal(store.getSnapshot().recognitions.length, 0)
})

test('association explicitly confirms same plant and data deletion needs confirmation', async () => {
  const calls: Array<{ path: string; data: unknown }> = []
  const store = new RemoteJournal(async (path, data) => {
    if (!data) return snapshot('alice')
    calls.push({ path, data })
    return { item: {}, snapshot: snapshot('alice') }
  }, () => {})
  await store.connect('alice')
  await store.linkRecognition('plant-one', 'record-one')
  await store.deleteAll()
  assert.deepEqual(calls, [
    { path: '/plants/plant-one/observations', data: {
      recognitionId: 'record-one', requestId: 'link:record-one', confirmedSamePlant: true,
      note: '由我确认，这是同一株植物。',
    } },
    { path: '/journal/clear', data: { confirm: true } },
  ])
})

test('stale refresh cannot overwrite a newer mutation snapshot', async () => {
  const delayed = deferred<ServerSnapshot>()
  let reads = 0
  const updated = snapshot('alice')
  const newRecord: Recognition = { ...updated.recognitions[0], id: 'new-record' }
  updated.recognitions = [newRecord]
  const store = new RemoteJournal(async (_path, data) => {
    if (data) return { item: newRecord, snapshot: updated }
    return ++reads === 1 ? snapshot('alice') : delayed.promise
  }, () => {})
  await store.connect('alice')
  const loading = store.refresh()
  await store.startRecognition('photo', 'request')
  delayed.resolve(snapshot('alice'))
  await loading
  assert.equal(store.getSnapshot().recognitions[0].id, 'new-record')
})

test('explicit species confirmation and clearing preserve original candidates', async () => {
  const state = snapshot('alice')
  const original = {
    source: 'BioCLIP', version: 'test-revision', isDemo: false,
    candidates: [{ speciesId: 'species-first', confidence: null }, { speciesId: 'species-other', confidence: null }],
  }
  state.recognitions = [{ ...state.recognitions[0], rawResult: original }]
  const calls: unknown[] = []
  const store = new RemoteJournal(async (path, data) => {
    if (!data) return state
    assert.equal(path, '/recognitions/record-alice/confirm')
    calls.push(data)
    state.recognitions = [{ ...state.recognitions[0], confirmedSpeciesId: data.speciesId as string | null }]
    return { item: state.recognitions[0], snapshot: state }
  }, () => {})
  await store.connect('alice')
  assert.equal(store.getSnapshot().recognitions[0].confirmedSpeciesId, null)
  await store.confirmSpecies('record-alice', 'species-first')
  assert.equal(store.getSnapshot().recognitions[0].confirmedSpeciesId, 'species-first')
  await store.confirmSpecies('record-alice', 'species-other')
  assert.equal(store.getSnapshot().recognitions[0].confirmedSpeciesId, 'species-other')
  await store.confirmSpecies('record-alice', null)
  assert.equal(store.getSnapshot().recognitions[0].confirmedSpeciesId, null)
  assert.equal(store.getSnapshot().recognitions[0].rawResult, original)
  assert.deepEqual(calls, [{ speciesId: 'species-first' }, { speciesId: 'species-other' }, { speciesId: null }])
})

test('confirmation applies the linked plant update even without an observation', async () => {
  const state = snapshot('alice')
  state.recognitions = [{ ...state.recognitions[0], linkedPlantId: 'stable-plant' }]
  state.plants = [{
    id: 'stable-plant', ownerId: 'alice', nickname: '窗边的植物', speciesId: null,
    locationDescription: '', coordinates: null, story: '', cover: state.recognitions[0].photo,
    createdAt: '2026-09-20T10:00:00Z',
  }]
  const store = new RemoteJournal(async (_path, data) => {
    if (!data) return state
    const speciesId = data.speciesId as string | null
    return {
      item: { ...state.recognitions[0], confirmedSpeciesId: speciesId },
      snapshot: {
        ...state,
        plants: [{ ...state.plants[0], speciesId }],
        recognitions: [{ ...state.recognitions[0], confirmedSpeciesId: speciesId }],
      },
    }
  }, () => {})
  await store.connect('alice')
  for (const speciesId of ['species-first', 'species-other', null]) {
    await store.confirmSpecies('record-alice', speciesId)
    assert.equal(store.getSnapshot().plants[0].speciesId, speciesId)
    assert.equal(store.getSnapshot().plants[0].id, 'stable-plant')
    assert.equal(store.getSnapshot().recognitions[0].linkedPlantId, 'stable-plant')
    assert.deepEqual(store.getSnapshot().observations, [])
  }
})

test('plant descriptions are sent independently of story and never as public species knowledge', async () => {
  const calls: unknown[] = []
  const store = new RemoteJournal(async (_path, data) => {
    if (!data) return snapshot('alice')
    calls.push(data)
    return { item: {}, snapshot: snapshot('alice') }
  }, () => {})
  await store.connect('alice')
  await store.createPlant('record-alice', { nickname: '这一株', location: '', story: '故事', description: '叶片描述' })
  await store.editPlant('plant', { nickname: '这一株', location: '', story: '故事', description: '', speciesId: null })
  assert.deepEqual(calls, [
    { recognitionId: 'record-alice', nickname: '这一株', location: '', story: '故事', description: '叶片描述' },
    { nickname: '这一株', location: '', story: '故事', description: '', speciesId: null },
  ])
})
