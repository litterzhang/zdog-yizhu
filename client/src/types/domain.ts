export type Id = string
export type RecognitionStatus = 'running' | 'succeeded' | 'uncertain' | 'failed'
export type MockScenario = 'success' | 'uncertain' | 'failure'

export interface Species {
  id: Id
  commonName: string
  isVisible?: boolean
  chineseName?: string | null
  chineseAliases?: string[]
  chineseNameOrigin?: 'colchina' | null
  nameSources?: { title: string; url: string; version?: string; sourceName?: string; matchType?: string; acceptancePolicy?: string; recordId?: string }[]
  scientificName: string
  /** Public species reference photo supplied by the server, not a private observation. */
  sampleImageUrl?: string | null
  taxonomy: null | { kingdom?: string; phylum?: string; class?: string; order?: string; family?: string; genus?: string }
  taxonomyChinese?: NonNullable<Species['taxonomy']>
  taxonomyAlternatives?: NonNullable<Species['taxonomy']>[]
  englishName?: string
  growthForms?: string[]
  growthFormSources?: { title: string; url: string }[]
  sources?: { title: string; url: string; version?: string }[]
  knowledge: {
    summary: string
    sources: { title: string; url: string }[]
    isAiGenerated?: boolean
    reviewStatus?: 'unreviewed' | 'reviewed'
    updatedAt?: string
  } | null
}

export interface Photo {
  id: Id
  uri: string
  origin: 'demo' | 'local' | 'remote'
}

export interface Candidate {
  speciesId: Id
  confidence: number | null
  score?: number
}

export interface ProviderResult {
  source: string
  version: string
  isDemo: boolean
  candidates: readonly Candidate[]
  catalogVersion?: string
  scoreType?: string
  reason?: string | null
}

export interface Recognition {
  id: Id
  ownerId: Id
  photo: Photo
  status: RecognitionStatus
  rawResult: ProviderResult | null
  confirmedSpeciesId: Id | null
  linkedPlantId?: Id | null
  associationConflict?: boolean
  createdAt: string
  failure: string | null
}

export interface Plant {
  id: Id
  ownerId: Id
  nickname: string
  speciesId: Id | null
  locationDescription: string
  coordinates: { latitude: number; longitude: number } | null
  story: string
  description?: string
  cover: Photo
  createdAt: string
}

export interface Observation {
  id: Id
  plantId: Id
  ownerId: Id
  observedAt: string
  photo: Photo
  note: string
  recognitionId: Id | null
}

export interface JournalSnapshot {
  plants: readonly Plant[]
  observations: readonly Observation[]
  recognitions: readonly Recognition[]
  loading?: boolean
  error?: string
  engine?: 'warming' | 'ready' | 'failed'
  catalog?: { available: boolean; plantCount: number; version: string; source: string }
}
