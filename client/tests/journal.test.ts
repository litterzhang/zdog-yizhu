import assert from 'node:assert/strict'
import { test } from 'node:test'
import { demoOwner, demoPhoto, demoPlantId, initialSnapshot } from '../src/services/fixtures'
import { MockJournal } from '../src/services/mock-journal'
import { MockRecognitionProvider, type RecognitionProvider } from '../src/services/recognition'
import type { ProviderResult } from '../src/types/domain'

const settle = () => new Promise<void>((resolve) => setImmediate(resolve))
const input = { nickname: '窗边的植物', location: '', story: '' }

test('mock provider labels demo data and never invents confidence', async () => {
  const result = await new MockRecognitionProvider('success', 0).recognize(demoPhoto)
  assert.equal(result.isDemo, true)
  assert.equal(result.source, 'MockRecognitionProvider')
  assert.equal(result.version, 'demo-v1')
  assert.ok(result.candidates.length > 1)
  assert.ok(result.candidates.every((candidate) => candidate.confidence === null))
})

test('task moves from running to success independently of page lifecycle', async () => {
  const store = new MockJournal()
  const id = store.startRecognition(demoPhoto, new MockRecognitionProvider('success', 0))
  assert.equal(store.getSnapshot().recognitions.find((item) => item.id === id)?.status, 'running')
  await settle()
  assert.equal(store.getSnapshot().recognitions.find((item) => item.id === id)?.status, 'succeeded')
})

test('failed task retries under the same recognition ID without duplication', async () => {
  const store = new MockJournal()
  const id = store.startRecognition(demoPhoto, new MockRecognitionProvider('failure', 0))
  await settle()
  assert.equal(store.getSnapshot().recognitions[0].status, 'failed')
  assert.ok(store.getSnapshot().recognitions[0].failure)
  store.retryRecognition(id, new MockRecognitionProvider('success', 0))
  store.retryRecognition(id, new MockRecognitionProvider('success', 0))
  await settle()
  assert.equal(store.getSnapshot().recognitions[0].id, id)
  assert.equal(store.getSnapshot().recognitions[0].status, 'succeeded')
  assert.equal(store.getSnapshot().recognitions.filter((item) => item.id === id).length, 1)
})

test('uncertain recognition still creates a plant and first observation', async () => {
  const store = new MockJournal()
  const id = store.startRecognition(demoPhoto, new MockRecognitionProvider('uncertain', 0))
  await settle()
  assert.equal(store.getSnapshot().recognitions[0].status, 'uncertain')
  const plant = store.createPlant(id, input)
  assert.equal(plant.speciesId, null)
  assert.equal(plant.coordinates, null)
  assert.notEqual(plant.id, id)
  const observation = store.getSnapshot().observations[0]
  assert.equal(observation.plantId, plant.id)
  assert.equal(observation.recognitionId, id)
  assert.equal(observation.photo.id, demoPhoto.id)
})

test('user corrections never mutate original provider candidates', () => {
  const store = new MockJournal()
  const before = store.getSnapshot().recognitions[0]
  const original = structuredClone(before.rawResult)
  store.confirmSpecies(before.id, 'species-jasmine')
  assert.equal(store.getSnapshot().recognitions[0].confirmedSpeciesId, 'species-jasmine')
  assert.deepEqual(store.getSnapshot().recognitions[0].rawResult, original)
  store.confirmSpecies(before.id, null)
  assert.equal(store.getSnapshot().recognitions[0].confirmedSpeciesId, null)
  assert.throws(() => store.confirmSpecies(before.id, 'unverified-species'))
})

test('duplicate create is idempotent, while duplicate nicknames have distinct IDs', async () => {
  const store = new MockJournal()
  const first = store.createPlant('recognition-demo-001', input)
  const repeated = store.createPlant('recognition-demo-001', input)
  assert.equal(first.id, repeated.id)
  assert.equal(store.getSnapshot().observations.filter((item) => item.plantId === first.id).length, 1)
  const id = store.startRecognition(demoPhoto, new MockRecognitionProvider('success', 0))
  await settle()
  const second = store.createPlant(id, input)
  assert.equal(first.nickname, second.nickname)
  assert.notEqual(first.id, second.id)
  assert.equal(second.speciesId, null, 'a candidate is not a confirmed species')
})

test('linking is explicit, idempotent, and does not replace plant species', () => {
  const store = new MockJournal()
  store.confirmSpecies('recognition-demo-001', 'species-jasmine')
  const first = store.linkRecognition(demoPlantId, 'recognition-demo-001')
  const second = store.linkRecognition(demoPlantId, 'recognition-demo-001')
  assert.equal(first.id, second.id)
  assert.equal(store.getSnapshot().plants[0].speciesId, 'species-ginkgo')
})

test('appending observations preserves the original plant ID and plant count', () => {
  const store = new MockJournal()
  const before = store.getSnapshot().plants
  const observation = store.appendObservation(demoPlantId, demoPhoto, '新的一天')
  assert.equal(observation.plantId, demoPlantId)
  assert.deepEqual(store.getSnapshot().plants, before)
})

test('local mock separates owners; this is not a cloud authorization test', () => {
  const store = new MockJournal('another-demo-owner')
  assert.deepEqual(store.getSnapshot(), { plants: [], observations: [], recognitions: [] })
  assert.throws(() => store.createPlant('recognition-demo-001', input))
  assert.throws(() => store.appendObservation(demoPlantId, demoPhoto, 'not mine'))
  assert.throws(() => store.confirmSpecies('recognition-demo-001', null))
  assert.throws(() => store.linkRecognition(demoPlantId, 'recognition-demo-001'))
})

test('validation rejects blank or overlong text and creating while running', () => {
  const store = new MockJournal()
  assert.throws(() => store.createPlant('recognition-demo-001', { ...input, nickname: ' ' }))
  assert.throws(() => store.createPlant('recognition-demo-001', { ...input, nickname: '长'.repeat(41) }))
  assert.throws(() => store.createPlant('recognition-demo-001', { ...input, story: '长'.repeat(501) }))
  assert.throws(() => store.appendObservation(demoPlantId, demoPhoto, '长'.repeat(501)))
  const id = store.startRecognition(demoPhoto, new MockRecognitionProvider('success', 0))
  assert.throws(() => store.createPlant(id, input))
})

test('clearing session prevents an in-flight completion restoring private records', async () => {
  let finish: (value: ProviderResult) => void = () => {}
  const provider: RecognitionProvider = {
    source: 'test', version: 'demo-v1', isDemo: true,
    recognize: () => new Promise((resolve) => { finish = resolve }),
  }
  const store = new MockJournal()
  store.startRecognition(demoPhoto, provider)
  store.clear()
  finish({ source: 'test', version: 'demo-v1', isDemo: true, candidates: [] })
  await settle()
  assert.deepEqual(store.getSnapshot(), { plants: [], observations: [], recognitions: [] })
})

test('session clear publishes once and subscribers can unsubscribe', () => {
  const store = new MockJournal(demoOwner, initialSnapshot())
  let updates = 0
  const unsubscribe = store.subscribe(() => { updates += 1 })
  store.clear()
  unsubscribe()
  store.clear()
  assert.equal(updates, 1)
})

test('real providers are blocked in the visual sample', () => {
  const store = new MockJournal()
  const provider: RecognitionProvider = {
    source: 'not-configured', version: 'unknown', isDemo: false,
    recognize: async () => { throw new Error('must not be called') },
  }
  assert.throws(() => store.startRecognition(demoPhoto, provider))
})
