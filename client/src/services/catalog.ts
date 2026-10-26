import { species as demoSpecies } from './fixtures'
import type { Species } from '../types/domain'

let catalog: readonly Species[] = []
// Reference metadata is public; keep selected search results separate from the
// account snapshot so background journal refreshes cannot erase an edit draft.
const publicSpecies = new Map<string, Species>()
export function setCatalog(value: readonly Species[]) { catalog = value }
export function rememberSpecies(value: Species) {
  publicSpecies.delete(value.id)
  publicSpecies.set(value.id, value)
  if (publicSpecies.size > 200) publicSpecies.delete(publicSpecies.keys().next().value!)
}
export function allSpecies() { return catalog }
export function catalogQuery(params: Record<string, string | number | undefined>) {
  return Object.entries(params).flatMap(([key, value]) => value === undefined
    ? [] : [`${key}=${encodeURIComponent(value)}`]).join('&')
}
export function catalogRouteId(value: string) {
  // Taro H5 retains encoded query values, while other targets may decode them.
  try { return decodeURIComponent(value) } catch { return value }
}
export function findSpecies(id: string | null | undefined) {
  return catalog.find((item) => item.id === id) ?? (id ? publicSpecies.get(id) : undefined)
    ?? demoSpecies.find((item) => item.id === id)
}
