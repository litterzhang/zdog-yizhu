import type { JournalSnapshot, Observation, Plant, Recognition, Species } from '../types/domain'

export interface ServerSnapshot extends JournalSnapshot {
  species: readonly Species[]
  engine: 'warming' | 'ready' | 'failed'
}
type Request = (path: string, data?: Record<string, unknown>) => Promise<unknown>
const empty = (): JournalSnapshot => ({
  recognitions: [], plants: [], observations: [], loading: false, error: '',
})

export class RemoteJournal {
  private snapshot = empty()
  private owner: string | null = null
  private epoch = 0
  private revision = 0
  private mutating = false
  private active = true
  private timer: ReturnType<typeof setTimeout> | undefined
  private listeners = new Set<() => void>()

  constructor(private request: Request, private catalog: (items: readonly Species[]) => void) {}
  getSnapshot = () => this.snapshot
  subscribe = (listener: () => void) => {
    this.listeners.add(listener)
    return () => { this.listeners.delete(listener) }
  }
  private publish(value: JournalSnapshot) {
    this.snapshot = value
    this.listeners.forEach((listener) => listener())
  }
  clear() {
    this.pause()
    this.epoch++
    this.owner = null
    this.mutating = false
    this.catalog([])
    this.publish(empty())
  }
  pause() {
    this.active = false
    clearTimeout(this.timer)
  }
  async connect(owner: string) {
    if (this.owner !== owner) { this.clear(); this.owner = owner }
    this.active = true
    await this.refresh()
  }
  private apply(value: ServerSnapshot) {
    if (!value || !Array.isArray(value.species) || !Array.isArray(value.plants)
      || !Array.isArray(value.recognitions) || !Array.isArray(value.observations)) {
      throw new Error('记录响应不完整，请重试。')
    }
    if ([...value.plants, ...value.recognitions, ...value.observations]
      .some((item) => item.ownerId !== this.owner)) throw new Error('记录身份校验未通过。')
    this.catalog(value.species)
    this.publish({ ...value, loading: false, error: '' })
    clearTimeout(this.timer)
    if (this.active && value.recognitions.some((item) => item.status === 'running')) {
      this.timer = setTimeout(() => { void this.refresh(true) }, 5000)
    }
  }
  async refresh(silent = false) {
    if (!this.owner) return
    const epoch = this.epoch
    const revision = ++this.revision
    if (!silent) this.publish({ ...this.snapshot, loading: true, error: '' })
    try {
      const value = await this.request('/journal') as ServerSnapshot
      if (epoch === this.epoch && revision === this.revision) this.apply(value)
    } catch (error) {
      if (epoch === this.epoch && revision === this.revision) {
        this.publish({ ...this.snapshot, loading: false, error: error instanceof Error ? error.message : '记录加载失败，请重试。' })
      }
    }
  }
  private async mutate<T>(path: string, data: Record<string, unknown>): Promise<T> {
    if (!this.owner) throw new Error('请先登录再保存记录。')
    if (this.mutating) throw new Error('上一项操作尚未完成，请稍候。')
    const epoch = this.epoch
    ++this.revision
    this.mutating = true
    try {
      const value = await this.request(path, data) as { item: T; snapshot: ServerSnapshot }
      if (epoch !== this.epoch) throw new Error('账号状态已变化，请重新操作。')
      ++this.revision
      this.apply(value.snapshot)
      return value.item
    } finally {
      if (epoch === this.epoch) this.mutating = false
    }
  }
  startRecognition(photoId: string, requestId: string) {
    return this.mutate<Recognition>('/recognitions', { photoId, requestId })
  }
  retryRecognition(id: string) { return this.mutate<Recognition>(`/recognitions/${encodeURIComponent(id)}/retry`, {}) }
  confirmSpecies(id: string, speciesId: string | null) {
    return this.mutate<Recognition>(`/recognitions/${encodeURIComponent(id)}/confirm`, { speciesId })
  }
  createPlant(recognitionId: string, input: { nickname: string; location: string; story: string; description?: string }) {
    return this.mutate<Plant>('/plants', { recognitionId, ...input })
  }
  editPlant(id: string, input: { nickname: string; location: string; story: string; speciesId: string | null; description?: string }) {
    return this.mutate<Plant>(`/plants/${encodeURIComponent(id)}/edit`, input)
  }
  appendObservation(plantId: string, photoId: string, note: string, requestId: string) {
    return this.mutate<Observation>(`/plants/${encodeURIComponent(plantId)}/observations`, { photoId, note, requestId })
  }
  linkRecognition(plantId: string, recognitionId: string) {
    return this.mutate<Observation>(`/plants/${encodeURIComponent(plantId)}/observations`, {
      recognitionId, requestId: `link:${recognitionId}`, confirmedSamePlant: true, note: '由我确认，这是同一株植物。',
    })
  }
  editObservation(id: string, note: string) {
    return this.mutate<Observation>(`/observations/${encodeURIComponent(id)}/edit`, { note })
  }
  remove(kind: 'plants' | 'recognitions' | 'observations', id: string) {
    return this.mutate<null>(`/journal/${kind}/${encodeURIComponent(id)}/delete`, {})
  }
  deleteAll() { return this.mutate<null>('/journal/clear', { confirm: true }) }
}
