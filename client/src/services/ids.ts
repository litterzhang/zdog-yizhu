let sequence = 0
const session = `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`

// Operation keys only; persistent entity IDs are assigned by Flask.
export function createLocalId(kind: string) {
  return `${kind}-${session}-${++sequence}`
}
