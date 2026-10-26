import { useSyncExternalStore } from 'react'
import { accountSession } from '../services/session'

export function useAccount() {
  return useSyncExternalStore(accountSession.subscribe, accountSession.getSnapshot, accountSession.getSnapshot)
}
