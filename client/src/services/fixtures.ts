import type { JournalSnapshot, Photo, Species } from '../types/domain'

export const demoPhoto: Photo = { id: 'photo-demo-leaf', uri: '/assets/photos/ginkgo-leaf.jpg', origin: 'demo' }
export const demoTreePhoto: Photo = { id: 'photo-demo-tree', uri: '/assets/photos/ginkgo-tree.jpg', origin: 'demo' }
export const demoOwner = 'local-demo-user'
export const demoPlantId = 'plant-demo-001'

export const species: readonly Species[] = [
  {
    id: 'species-ginkgo',
    commonName: '银杏',
    scientificName: 'Ginkgo biloba',
    taxonomy: null,
    knowledge: null,
  },
  {
    id: 'species-jasmine',
    commonName: '茉莉',
    scientificName: 'Jasminum sambac',
    taxonomy: null,
    knowledge: null,
  },
]

export function initialSnapshot(): JournalSnapshot {
  return {
    plants: [{
      id: demoPlantId,
      ownerId: demoOwner,
      nickname: '回家路上的银杏',
      speciesId: 'species-ginkgo',
      locationDescription: '巷口，转弯的地方',
      coordinates: null,
      story: '以前总是匆匆经过。今天停下来，才发现阳光穿过叶子的样子很好看。',
      cover: demoTreePhoto,
      createdAt: '2026-09-12T08:30:00+08:00',
    }],
    observations: [{
      id: 'observation-demo-002',
      plantId: demoPlantId,
      ownerId: demoOwner,
      observedAt: '2026-09-18T16:20:00+08:00',
      photo: demoPhoto,
      note: '又来看它。抬头的时候，刚好有一阵风。',
      recognitionId: 'recognition-demo-001',
    }, {
      id: 'observation-demo-001',
      plantId: demoPlantId,
      ownerId: demoOwner,
      observedAt: '2026-09-12T08:30:00+08:00',
      photo: demoTreePhoto,
      note: '第一次认真记下这株植物。',
      recognitionId: null,
    }],
    recognitions: [{
      id: 'recognition-demo-001',
      ownerId: demoOwner,
      photo: demoPhoto,
      status: 'succeeded',
      rawResult: {
        source: 'MockRecognitionProvider',
        version: 'demo-v1',
        isDemo: true,
        candidates: [
          { speciesId: 'species-ginkgo', confidence: null },
          { speciesId: 'species-jasmine', confidence: null },
        ],
      },
      confirmedSpeciesId: null,
      createdAt: '2026-09-18T16:20:00+08:00',
      failure: null,
    }],
  }
}
