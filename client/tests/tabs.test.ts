import assert from 'node:assert/strict'
import { test } from 'node:test'
import { navigationTabs } from '../src/config/navigation'
import { initialSnapshot } from '../src/services/fixtures'
import { catalogQuery, catalogRouteId, findSpecies, rememberSpecies, setCatalog } from '../src/services/catalog'
import { catalogDetailSources, groupRecognitions, lastObservationAt, recognitionCandidates, recognitionPreview, recognitionTitle, searchPlants, speciesSampleImageUrl, speciesSearchName } from '../src/services/presentation'

test('navigation has the four requested tabs, with distinct pages', () => {
  assert.deepEqual(navigationTabs.map((tab) => tab.text), ['发现', '足迹', '株记', '我的'])
  assert.equal(new Set(navigationTabs.map((tab) => tab.pagePath)).size, 4)
})

test('plant search matches nickname and species without changing saved plants', () => {
  const { plants } = initialSnapshot()
  assert.equal(searchPlants(plants, '回家路上').length, 1)
  assert.equal(searchPlants(plants, '银杏').length, 1)
  assert.equal(searchPlants(plants, ' GINKGO ').length, 1)
  assert.equal(searchPlants(plants, '未记录的植物').length, 0)
  assert.equal(searchPlants(plants, '').length, 1)
  assert.equal(plants[0].nickname, '回家路上的银杏')
})

test('footprints are ordered by recognition date without mutating source records', () => {
  const sample = initialSnapshot().recognitions[0]
  const records = [
    { ...sample, id: 'earlier', createdAt: '2026-09-18T10:00:00+08:00' },
    { ...sample, id: 'latest', createdAt: '2026-09-20T10:00:00+08:00' },
    { ...sample, id: 'same-day', createdAt: '2026-09-20T09:00:00+08:00' },
  ]
  const groups = groupRecognitions(records)
  assert.equal(groups.length, 2)
  assert.deepEqual(groups[0].records.map((item) => item.id), ['latest', 'same-day'])
  assert.equal(records[0].id, 'earlier')
})

test('history labels distinguish unknown, failed and manually selected results', () => {
  const record = initialSnapshot().recognitions[0]
  assert.equal(recognitionTitle(record), '银杏')
  assert.equal(recognitionTitle({ ...record, confirmedSpeciesId: 'species-jasmine' }), '茉莉')
  assert.equal(recognitionTitle({ ...record, status: 'uncertain' }), '名字，留待下次确认')
  assert.equal(recognitionTitle({ ...record, status: 'failed' }), '这次还没有认出来')
})

test('saved plant dates reflect the latest observation and do not create new plants', () => {
  const snapshot = initialSnapshot()
  assert.equal(lastObservationAt(snapshot.plants[0], snapshot), '2026-09-18T16:20:00+08:00')
  assert.equal(lastObservationAt(snapshot.plants[0], { ...snapshot, observations: [] }), snapshot.plants[0].createdAt)
})

test('public species selected while editing survives a private journal refresh', () => {
  const selected = {
    id: 'tol10m:test-new-species', commonName: 'Synthetic species', scientificName: 'Synthetic species',
    taxonomy: { order: 'Testales', family: 'Testaceae', genus: 'Synthetic' }, knowledge: null,
  }
  rememberSpecies(selected)
  setCatalog([])
  assert.equal(findSpecies(selected.id), selected)
})

test('catalog requests encode search input and omit absent parent IDs', () => {
  assert.equal(catalogQuery({ q: '银杏 & %', parent: undefined, offset: 0 }),
    `q=${encodeURIComponent('银杏 & %')}&offset=0`)
})

test('species deep links support encoded Taro H5 queries and decoded mini-program queries', () => {
  assert.equal(catalogRouteId('tol10m%3Aabc'), 'tol10m:abc')
  assert.equal(catalogRouteId('bioclip:plant-001'), 'bioclip:plant-001')
  assert.equal(catalogRouteId('%invalid'), '%invalid')
})

test('fresh Chinese reference names replace Latin display labels without changing species IDs', () => {
  const old = {
    id: 'tol10m:sunflower-test', commonName: 'Helianthus annuus',
    scientificName: 'Helianthus annuus', taxonomy: null, knowledge: null,
  }
  rememberSpecies(old)
  setCatalog([{ ...old, commonName: '向日葵', chineseName: '向日葵' }])
  assert.equal(findSpecies(old.id)?.commonName, '向日葵')
  assert.equal(findSpecies(old.id)?.scientificName, old.scientificName)
  setCatalog([])
})

test('hidden historical candidates stay stored without promoting the next candidate to first', () => {
  const hidden = {
    id: 'tol10m:hidden', commonName: 'Hidden plant', scientificName: 'Hidden plant',
    isVisible: false, taxonomy: null, knowledge: null,
  }
  const visible = { ...hidden, id: 'tol10m:visible', commonName: '示例草', isVisible: true }
  setCatalog([hidden, visible])
  const base = initialSnapshot().recognitions[0]
  const record = {
    ...base, confirmedSpeciesId: null,
    rawResult: { ...base.rawResult!, candidates: [
      { speciesId: hidden.id, confidence: null }, { speciesId: visible.id, confidence: null },
    ] },
  }
  assert.equal(recognitionCandidates(record).hiddenLeading, true)
  assert.deepEqual(recognitionCandidates(record).candidates, [record.rawResult.candidates[1]])
  assert.equal(recognitionTitle(record), '名字，留待下次确认')
  assert.equal(record.rawResult.candidates.length, 2)
  assert.equal(recognitionTitle({ ...record, confirmedSpeciesId: hidden.id }), 'Hidden plant')
  assert.equal(findSpecies(hidden.id), hidden) // Keep saved private references readable.
  assert.equal(recognitionPreview(record, null).species, undefined)
  assert.equal(recognitionPreview(record, { recognitionId: record.id, speciesId: hidden.id }).species, undefined)
  assert.equal(recognitionPreview(record, { recognitionId: record.id, speciesId: visible.id }).species?.id, visible.id)
  setCatalog([])
})

test('candidate preview changes the displayed species without confirming or rewriting original results', () => {
  const record = initialSnapshot().recognitions[0]
  const original = structuredClone(record)
  assert.equal(recognitionPreview(record, null).species?.id, 'species-ginkgo')
  const preview = recognitionPreview(record, { recognitionId: record.id, speciesId: 'species-jasmine' })
  assert.equal(preview.species?.commonName, '茉莉')
  assert.equal(preview.species?.scientificName, 'Jasminum sambac')
  assert.equal(preview.species?.id, 'species-jasmine')
  assert.equal(preview.isConfirmed, false)
  assert.deepEqual(record, original)
})

test('previewing another candidate does not replace the saved confirmation until explicitly confirmed', () => {
  const record = { ...initialSnapshot().recognitions[0], confirmedSpeciesId: 'species-ginkgo' }
  const selection = { recognitionId: record.id, speciesId: 'species-jasmine' }
  assert.equal(recognitionPreview(record, null).isConfirmed, true)
  assert.equal(recognitionPreview(record, selection).isConfirmed, false)
  assert.equal(record.confirmedSpeciesId, 'species-ginkgo')
  assert.equal(recognitionPreview({ ...record, confirmedSpeciesId: selection.speciesId }, null).isConfirmed, true)
  assert.equal(recognitionPreview({ ...record, confirmedSpeciesId: null }, null).species?.id, 'species-ginkgo')
})

test('candidate previews cannot leak to other records or invent candidates', () => {
  const record = initialSnapshot().recognitions[0]
  assert.equal(recognitionPreview(record, { recognitionId: 'another-record', speciesId: 'species-jasmine' }).species?.id, 'species-ginkgo')
  const single = { ...record, rawResult: { ...record.rawResult!, candidates: record.rawResult!.candidates.slice(0, 1) } }
  assert.equal(recognitionPreview(single, { recognitionId: record.id, speciesId: 'species-jasmine' }).species?.id, 'species-ginkgo')
  assert.equal(recognitionPreview(undefined, null).species, undefined)
})

test('uncertain results only preview a candidate after explicit selection', () => {
  const record = { ...initialSnapshot().recognitions[0], status: 'uncertain' as const }
  assert.equal(recognitionPreview(record, null).species, undefined)
  assert.equal(recognitionPreview(record, { recognitionId: record.id, speciesId: 'species-jasmine' }).species?.id, 'species-jasmine')
  assert.equal(record.confirmedSpeciesId, null)
})

test('catalog selection previews a non-candidate without changing saved confirmation or raw results', () => {
  const species = {
    id: 'catalog:preview-test', commonName: '搜索物种', scientificName: 'Search species',
    taxonomy: null, knowledge: null,
  }
  rememberSpecies(species)
  const record = { ...initialSnapshot().recognitions[0], confirmedSpeciesId: 'species-ginkgo' }
  const original = structuredClone(record)
  const selection = { recognitionId: record.id, speciesId: species.id, source: 'catalog' as const }
  const preview = recognitionPreview(record, selection)
  assert.equal(preview.species, species)
  assert.equal(preview.isConfirmed, false)
  assert.deepEqual(record, original)
  assert.equal(recognitionPreview(record, { ...selection, source: 'candidate' }).species?.id, 'species-ginkgo')
  assert.equal(recognitionPreview(record, { ...selection, recognitionId: 'another-record' }).species?.id, 'species-ginkgo')
  assert.equal(recognitionPreview({ ...record, confirmedSpeciesId: species.id }, null).isConfirmed, true)
  setCatalog([])
  assert.equal(recognitionPreview(record, selection).species, species)
})

test('catalog preview works without recognition candidates and clears back to the saved result', () => {
  const record = {
    ...initialSnapshot().recognitions[0], status: 'uncertain' as const,
    rawResult: null, confirmedSpeciesId: null,
  }
  const selection = { recognitionId: record.id, speciesId: 'species-jasmine', source: 'catalog' as const }
  assert.equal(recognitionPreview(record, selection).species?.id, 'species-jasmine')
  assert.equal(recognitionPreview(record, selection).isConfirmed, false)
  assert.equal(recognitionPreview(record, null).species, undefined)
  assert.equal(record.rawResult, null)
})

test('catalog preview does not display unknown or newly hidden search selections', () => {
  const hidden = {
    id: 'catalog:hidden-preview', commonName: '已隐藏', scientificName: 'Hidden species',
    isVisible: false, taxonomy: null, knowledge: null,
  }
  setCatalog([hidden])
  const record = initialSnapshot().recognitions[0]
  for (const speciesId of [hidden.id, 'catalog:missing-preview']) {
    assert.equal(recognitionPreview(record, {
      recognitionId: record.id, speciesId, source: 'catalog',
    }).species?.id, 'species-ginkgo')
  }
  setCatalog([])
})

test('species search prefers accepted Chinese aliases without changing the catalog object', () => {
  const source = (sourceName: string) => ({
    title: 'Test source', url: 'https://example.org/source', sourceName,
    acceptancePolicy: 'visible-sourced-aliases-v1', matchType: 'exact', recordId: 'test:1',
  })
  const item = {
    id: 'catalog:alias-test', commonName: '名录名称', chineseName: '名录名称',
    scientificName: 'Example species', taxonomy: null, knowledge: null,
    chineseAliases: ['待核别名', '常用别名', '其他别名'],
    nameSources: [source('其他别名'), source('常用别名')],
  }
  const original = structuredClone(item)
  assert.equal(speciesSearchName(item), '常用别名')
  assert.deepEqual(item, original)
  assert.equal(speciesSearchName({ ...item, isVisible: false }), '名录名称')
  for (const nameSources of [
    [],
    [{ ...source('常用别名'), acceptancePolicy: 'pending' }],
    [{ ...source('常用别名'), matchType: 'fuzzy' }],
    [{ ...source('常用别名'), recordId: '' }],
    [{ ...source('常用别名'), url: '' }],
  ]) {
    assert.equal(speciesSearchName({ ...item, nameSources }), '名录名称')
  }
})

test('species search falls back through Chinese name and existing name without empty or Latin aliases', () => {
  const item = {
    id: 'catalog:alias-fallback', commonName: 'Existing name', chineseName: '中文名称',
    scientificName: 'Example species', taxonomy: null, knowledge: null,
    chineseAliases: ['', ' ', 'English alias'],
    nameSources: ['', ' ', 'English alias'].map((sourceName) => ({
      title: 'Test source', url: 'https://example.org/source', sourceName,
      acceptancePolicy: 'visible-sourced-aliases-v1', matchType: 'exact', recordId: 'test:1',
    })),
  }
  assert.equal(speciesSearchName(item), '中文名称')
  assert.equal(speciesSearchName({ ...item, chineseName: null }), 'Existing name')
  assert.equal(speciesSearchName({ ...item, chineseName: null, commonName: '' }), 'Example species')
})

test('catalog details show each source URL once and omit growth-only references without mutating data', () => {
  const source = { title: 'Imageomics / TreeOfLife-10M', url: 'https://example.org/catalog' }
  const other = { title: 'Other reference', url: 'https://example.org/other' }
  const item = {
    id: 'catalog:sources-test', commonName: '测试植物', scientificName: 'Example species',
    taxonomy: null, knowledge: null, growthForms: ['tree'],
    sources: [source, { ...source }, other],
    growthFormSources: [source, { title: 'Growth reference', url: 'https://example.org/growth' }],
  }
  const original = structuredClone(item)
  assert.deepEqual(catalogDetailSources(item), [source, other])
  assert.deepEqual(item, original)
  assert.deepEqual(catalogDetailSources({ ...item, sources: undefined }), [])
})

test('catalog details do not repeat credits already displayed with species knowledge', () => {
  const source = { title: 'Shared source', url: 'https://example.org/shared' }
  const other = { title: 'Shared source', url: 'https://example.org/distinct' }
  const item = {
    id: 'catalog:knowledge-sources', commonName: '测试植物', scientificName: 'Example species',
    taxonomy: null, sources: [source, other],
    knowledge: { summary: '有来源的简介。', sources: [source] },
  }
  assert.deepEqual(catalogDetailSources(item), [other])
  assert.equal(item.knowledge.sources.length, 1)
})

test('species sample image uses only the server-provided HTTPS URL without changing the species', () => {
  const item = {
    id: 'catalog:sample-image', commonName: '测试植物', scientificName: 'Example species',
    taxonomy: null, knowledge: null,
    sampleImageUrl: '  https://images.example.org/plants/reference.jpg?size=medium&v=2  ',
  }
  const original = structuredClone(item)
  assert.equal(speciesSampleImageUrl(item), 'https://images.example.org/plants/reference.jpg?size=medium&v=2')
  assert.deepEqual(item, original)
})

test('missing or invalid sample URLs never produce fallback or fabricated images', () => {
  const base = {
    id: 'catalog:missing-image', commonName: '银杏', scientificName: 'Ginkgo biloba',
    taxonomy: null, knowledge: null,
  }
  assert.equal(speciesSampleImageUrl(base), null)
  assert.equal(speciesSampleImageUrl(undefined), null)
  assert.equal(speciesSampleImageUrl(null), null)
  for (const sampleImageUrl of [
    null, '', '  ', 'http://images.example.org/a.jpg', '/assets/photos/ginkgo-leaf.jpg',
    '//images.example.org/a.jpg', 'data:image/png;base64,abc', 'javascript:alert(1)',
    'https://', 'https://images.example.org/a b.jpg', 'https://user:secret@images.example.org/a.jpg',
  ]) {
    assert.equal(speciesSampleImageUrl({ ...base, sampleImageUrl }), null)
  }
})
