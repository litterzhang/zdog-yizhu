export interface AccountUser {
  id: string
  nickname: string
}

export class AuthError extends Error {
  constructor(readonly status: number, message: string) {
    super(message)
    this.name = 'AuthError'
  }
}

interface AuthRequest {
  path: '/wechat' | '/login' | '/me'
  method: 'POST' | 'GET'
  data?: Record<string, string>
  token?: string
}

export type AuthTransport = (request: AuthRequest) => Promise<{ statusCode: number; data: unknown }>

function record(value: unknown): Record<string, unknown> {
  return value && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : {}
}

export function validToken(value: unknown): value is string {
  return typeof value === 'string' && value.length > 0 && value.length <= 8192 && !/\s/.test(value)
}

export class AuthApi {
  constructor(private readonly transport: AuthTransport) {}

  private async request(input: AuthRequest): Promise<Record<string, unknown>> {
    let response: Awaited<ReturnType<AuthTransport>>
    try {
      response = await this.transport(input)
    } catch {
      // Never display arbitrary transport messages (may include private payloads).
      throw new AuthError(0, '无法连接登录服务。请检查网络后重试；小程序需配置 API 请求合法域名。')
    }
    if (response.statusCode !== 200) {
      const messages: Record<number, string> = {
        400: '登录信息不正确，请检查后重试。',
        401: input.path === '/me' ? '登录已过期，请重新登录。' : '登录未通过，请检查账号密码或重新尝试微信登录。',
        403: '当前登录请求不被允许，请联系维护者检查认证配置。',
        429: '登录操作过于频繁，请一分钟后重试。',
        502: '认证服务响应异常，请稍后重试。',
        503: '登录服务暂不可用，请稍后重试或联系维护者检查认证配置。',
        504: '登录服务超时，请重试。',
      }
      throw new AuthError(response.statusCode, messages[response.statusCode] || '登录请求未完成，请稍后重试。')
    }
    const data = record(response.data)
    if (data.code !== 200) throw new AuthError(502, '登录响应格式不正确，请稍后重试。')
    return data
  }

  async wechat(code: string): Promise<string> {
    if (!code.trim()) throw new AuthError(0, '没有取得微信登录凭证，请重试。')
    return this.token({ path: '/wechat', method: 'POST', data: { code } })
  }

  async password(username: string, password: string): Promise<string> {
    if (!username.trim() || username.trim().length > 128 || !password || password.length > 256) {
      throw new AuthError(0, '请填写完整的账号和密码。')
    }
    return this.token({ path: '/login', method: 'POST', data: { username: username.trim(), password } })
  }

  private async token(input: AuthRequest): Promise<string> {
    const data = await this.request(input)
    if (!validToken(data.access_token)) throw new AuthError(502, '登录响应缺少有效凭据，请重试。')
    return data.access_token
  }

  async me(token: string): Promise<AccountUser> {
    const data = await this.request({ path: '/me', method: 'GET', token })
    const user = record(data.user)
    if (typeof user.id !== 'string' || !user.id.trim()) throw new AuthError(502, '无法确认账号身份，请重试。')
    return {
      id: user.id,
      nickname: typeof user.nickname === 'string' && user.nickname.trim() ? user.nickname : '草木观察者',
    }
  }
}
