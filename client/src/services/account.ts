import { AuthError, type AuthApi, type AccountUser, validToken } from './auth-api'

export interface AccountSnapshot {
  status: 'guest' | 'loading' | 'authenticated' | 'error'
  user: AccountUser | null
  error: string
}

// UI visibility only; the backend still authorizes every private request.
export function hasVerifiedAccount(session: AccountSnapshot): boolean {
  return session.status === 'authenticated' && session.user !== null
}

export interface TokenStorage {
  read(): unknown
  write(token: string): void
  remove(): void
}

// Tokens never appear in React state, logs, URLs or journal ownership.
export class AccountStore {
  private snapshot: AccountSnapshot = { status: 'guest', user: null, error: '' }
  private readonly listeners = new Set<() => void>()
  private generation = 0
  private verifiedUserId: string | null = null

  constructor(
    private readonly api: AuthApi,
    private readonly storage: TokenStorage,
    private readonly resetPrivateState: (reason: 'login' | 'logout' | 'expired') => void,
  ) {}

  getSnapshot = () => this.snapshot
  subscribe = (listener: () => void) => {
    this.listeners.add(listener)
    return () => { this.listeners.delete(listener) }
  }

  private publish(next: AccountSnapshot) {
    this.snapshot = next
    this.listeners.forEach((listener) => listener())
  }

  private fail(error: unknown) {
    this.publish({
      status: 'error',
      user: null,
      error: error instanceof AuthError ? error.message : '登录操作未完成，请重试。',
    })
  }

  async signIn(consented: boolean, getToken: () => Promise<string>): Promise<void> {
    if (!consented) throw new AuthError(0, '请先阅读并同意登录与隐私说明。')
    if (this.snapshot.status === 'loading') throw new AuthError(0, '正在处理登录，请稍候。')
    const generation = ++this.generation
    this.publish({ status: 'loading', user: null, error: '' })
    try {
      // Discard any old credential before an explicit account switch.
      this.storage.remove()
      const token = await getToken()
      if (!validToken(token)) throw new AuthError(502, '登录响应不完整，请重试。')
      if (generation !== this.generation) throw new AuthError(0, '登录已取消。')
      const user = await this.api.me(token)
      if (generation !== this.generation) throw new AuthError(0, '登录已取消。')
      this.storage.write(token)
      this.verifiedUserId = user.id
      this.resetPrivateState('login')
      this.publish({ status: 'authenticated', user, error: '' })
    } catch (error) {
      if (generation === this.generation) this.fail(error)
      throw error instanceof AuthError ? error : new AuthError(0, '无法保存登录状态，请检查本机存储后重试。')
    }
  }

  // Saved tokens require server validation. Network failures retain the token
  // for retry, but never expose an authenticated user or create a demo account.
  async restore(): Promise<void> {
    if (this.snapshot.status === 'loading') return
    const generation = ++this.generation
    this.publish({ status: 'loading', user: null, error: '' })
    try {
      const token = this.storage.read()
      if (!validToken(token)) {
        if (token) this.storage.remove()
        if (this.verifiedUserId) {
          this.verifiedUserId = null
          this.resetPrivateState('expired')
        }
        this.publish({ status: 'guest', user: null, error: '' })
        return
      }
      const user = await this.api.me(token)
      if (generation !== this.generation) return
      if (this.verifiedUserId && this.verifiedUserId !== user.id) this.resetPrivateState('expired')
      this.verifiedUserId = user.id
      this.publish({ status: 'authenticated', user, error: '' })
    } catch (error) {
      if (generation !== this.generation) return
      if (error instanceof AuthError && error.status === 401) {
        this.verifiedUserId = null
        this.resetPrivateState('expired')
        try {
          this.storage.remove()
        } catch {
          this.fail(new AuthError(0, '会话已失效，但本机凭据未能清除，请重试退出登录。'))
          return
        }
        this.publish({ status: 'guest', user: null, error: '登录已过期，请重新登录。' })
      } else {
        this.fail(error)
      }
    }
  }

  signOut() {
    ++this.generation // In-flight login/restore must not resurrect a session.
    this.verifiedUserId = null
    this.resetPrivateState('logout')
    try {
      this.storage.remove()
      this.publish({ status: 'guest', user: null, error: '' })
    } catch {
      const error = new AuthError(0, '本机登录凭据未能清除，请重试退出登录。')
      this.fail(error)
      throw error
    }
  }
}
