import { RemoteJournal } from './remote-journal'
import { apiRequest } from './transport'
import { setCatalog } from './catalog'

export const journal = new RemoteJournal(apiRequest, setCatalog)
