import Taro from '@tarojs/taro'
import { apiBaseUrl } from '../config/api'
import { parseResponse } from './file-transfer'
import type { Species } from '../types/domain'
import { catalogQuery } from './catalog'

export interface CatalogInfo {
  available: boolean
  source: string
  version: string
  plantCount: number
  labelCount: number
  growthFormCount: number
  chineseNameCount?: number
  chineseTaxonCount?: number
  visiblePlantCount?: number
  hiddenPlantCount?: number
}
export interface TaxonNode { id: string; name: string; chineseName?: string | null; rank: 'order' | 'family' | 'genus'; count: number }
export interface CatalogPage<T> { items: T[]; total: number; offset: number; limit: number }
export const growthForms = [
  { id: '', name: '全部形态' }, { id: 'herb', name: '草本' }, { id: 'shrub', name: '灌木' },
  { id: 'tree', name: '乔木' }, { id: 'climber', name: '藤本' }, { id: 'unknown', name: '形态未知' },
]
export const rankNames = { order: '目', family: '科', genus: '属' }

async function read<T>(path: string): Promise<T> {
  let response: { statusCode: number; data: unknown }
  try {
    // Public reference data: no account, token, or private photographs are sent.
    response = await Taro.request({ url: `${apiBaseUrl}/catalog${path}`, method: 'GET', timeout: 20000 })
  } catch { throw new Error('名录暂时无法连接，请检查网络后重试。') }
  return parseResponse(response.statusCode, response.data) as T
}
export const catalogInfo = () => read<CatalogInfo>('')
export const searchSpecies = (params: { q: string; offset?: number; parent?: string; growthForm?: string; chineseOnly?: 0 | 1 }) =>
  read<CatalogPage<Species>>(`/species?${catalogQuery({ limit: 30, ...params })}`)
export const browseTaxa = (params: { parent?: string; growthForm?: string; offset?: number; chineseOnly?: 0 | 1 }) =>
  read<CatalogPage<TaxonNode>>(`/taxa?${catalogQuery({ limit: 30, ...params })}`)
export const readSpecies = (id: string) => read<Species>(`/species/${encodeURIComponent(id)}`)
