import { findSpecies } from './catalog'
import type { JournalSnapshot, Plant, Recognition, Species } from '../types/domain'

export function speciesSampleImageUrl(species?: Species | null): string | null {
  const value = species?.sampleImageUrl
  if (typeof value !== 'string') return null
  const url = value.trim()
  // Absolute HTTPS only; never derive an image from a name, ID or private photo.
  return /^https:\/\/[^/?#\s@]+(?:[/?#][^\s]*)?$/i.test(url) ? url : null
}

export function catalogDetailSources(species: Species) {
  // Knowledge already renders its own credits. Growth-form-only references
  // are not shown when the corresponding field is absent from this page.
  const seen = new Set(species.knowledge?.sources.map((source) => source.url))
  return (species.sources ?? []).filter((source) => {
    if (seen.has(source.url)) return false
    seen.add(source.url)
    return true
  })
}

export function speciesSearchName(species: Species): string {
  if (species.isVisible === false) return species.commonName
  // Match the server's journal display policy; do not promote pending aliases.
  const acceptedAliases = new Set(species.nameSources?.filter((source) =>
    source.acceptancePolicy === 'visible-sourced-aliases-v1'
    && source.matchType === 'exact' && source.recordId && source.url,
  ).map((source) => source.sourceName))
  const hasChinese = (name: string) => /[\u3400-\u4dbf\u4e00-\u9fff]/.test(name)
  const alias = species.chineseAliases?.find((name) => acceptedAliases.has(name) && hasChinese(name))
  if (alias) return alias.trim()
  if (species.chineseName && hasChinese(species.chineseName)) return species.chineseName.trim()
  return species.commonName || species.scientificName
}

export function recognitionCandidates(record: Recognition | undefined) {
  const original = record?.rawResult?.candidates ?? []
  return {
    candidates: original.filter((item) => findSpecies(item.speciesId)?.isVisible !== false),
    hiddenLeading: findSpecies(original[0]?.speciesId)?.isVisible === false,
  }
}

export interface CandidateSelection {
  recognitionId: string
  speciesId: string
  source?: 'candidate' | 'catalog'
}

// Candidate and catalog selections only preview a species; confirmation persists it.
export function recognitionPreview(record: Recognition | undefined, selection: CandidateSelection | null) {
  const { candidates, hiddenLeading } = recognitionCandidates(record)
  const selected = record && selection?.recognitionId === record.id
    ? findSpecies(selection.source === 'catalog'
      ? selection.speciesId
      : candidates.find((candidate) => candidate.speciesId === selection.speciesId)?.speciesId)
    : undefined
  const species = (selected?.isVisible !== false ? selected : undefined)
    ?? findSpecies(record?.confirmedSpeciesId)
    ?? (record?.status === 'succeeded' && !hiddenLeading ? findSpecies(candidates[0]?.speciesId) : undefined)
  return { species, isConfirmed: Boolean(species && species.id === record?.confirmedSpeciesId) }
}

export function speciesName(id: string | null | undefined): string {
  return findSpecies(id)?.commonName ?? '物种待确认'
}

export function recognitionTitle(record: Recognition): string {
  if (record.status === 'running') return '正在认识它'
  if (record.status === 'failed') return '这次还没有认出来'
  if (record.status === 'uncertain') return '名字，留待下次确认'
  if (!record.confirmedSpeciesId && recognitionCandidates(record).hiddenLeading) return '名字，留待下次确认'
  return speciesName(record.confirmedSpeciesId ?? record.rawResult?.candidates[0]?.speciesId)
}

export function recognitionState(record: Recognition): string {
  if (record.status === 'running') return '识别中'
  if (record.status === 'failed') return '可重试'
  if (record.status === 'uncertain') return '待确认'
  return record.confirmedSpeciesId ? '手动选择' : record.rawResult?.isDemo ? '演示候选' : '候选待确认'
}

export function dateLabel(iso: string): string {
  const date = new Date(iso)
  return `${date.getFullYear()}年${date.getMonth() + 1}月${date.getDate()}日`
}

export function timeLabel(iso: string): string {
  const date = new Date(iso)
  return `${String(date.getHours()).padStart(2, '0')}:${String(date.getMinutes()).padStart(2, '0')}`
}

export function groupRecognitions(records: readonly Recognition[]) {
  const groups: { date: string; records: Recognition[] }[] = []
  const sorted = [...records].sort((a, b) => Date.parse(b.createdAt) - Date.parse(a.createdAt))
  for (const record of sorted) {
    const date = dateLabel(record.createdAt)
    const group = groups[groups.length - 1]
    if (group?.date === date) group.records.push(record)
    else groups.push({ date, records: [record] })
  }
  return groups
}

export function searchPlants(plants: readonly Plant[], query: string): Plant[] {
  const search = query.trim().toLocaleLowerCase()
  return plants.filter((plant) => {
    const botanical = findSpecies(plant.speciesId)
    return [plant.nickname, botanical?.commonName, botanical?.scientificName]
      .some((name) => name?.toLocaleLowerCase().includes(search))
  })
}

export function lastObservationAt(plant: Plant, snapshot: JournalSnapshot): string {
  return snapshot.observations
    .filter((item) => item.plantId === plant.id)
    .map((item) => item.observedAt)
    .sort((a, b) => Date.parse(b) - Date.parse(a))[0] ?? plant.createdAt
}
