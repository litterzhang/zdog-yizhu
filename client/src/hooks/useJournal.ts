import { useSyncExternalStore } from 'react'
import { journal } from '../services/journal'

export function useJournal() {
  return useSyncExternalStore(journal.subscribe, journal.getSnapshot, journal.getSnapshot)
}
