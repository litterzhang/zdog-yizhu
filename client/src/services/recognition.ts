import type { MockScenario, Photo, ProviderResult } from '../types/domain'

// Real providers must run through an authenticated backend. No secrets in the app.
export interface RecognitionProvider {
  readonly source: string
  readonly version: string
  readonly isDemo: boolean
  recognize(photo: Photo): Promise<ProviderResult>
}

export class MockRecognitionProvider implements RecognitionProvider {
  readonly source = 'MockRecognitionProvider'
  readonly version = 'demo-v1'
  readonly isDemo = true

  constructor(
    private readonly scenario: MockScenario = 'success',
    private readonly delayMs = 900,
  ) {}

  async recognize(_photo: Photo): Promise<ProviderResult> {
    if (this.delayMs > 0) await new Promise((resolve) => setTimeout(resolve, this.delayMs))
    if (this.scenario === 'failure') throw new Error('演示识别失败。照片未上传，请重试或直接记录。')
    return {
      source: this.source,
      version: this.version,
      isDemo: this.isDemo,
      candidates: this.scenario === 'uncertain' ? [] : [
        { speciesId: 'species-ginkgo', confidence: null },
        { speciesId: 'species-jasmine', confidence: null },
      ],
    }
  }
}
