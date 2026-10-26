import Taro from '@tarojs/taro'
import { apiBaseUrl } from '../config/api'
import { AccountStore } from './account'
import { AuthApi, AuthError } from './auth-api'
import { journal } from './journal'
import { clearPhotoCache, setUnauthorizedHandler, tokenKey } from './transport'

export const authApi = new AuthApi(async ({ path, method, data, token }) => {
  const response = await Taro.request<unknown>({
    url: `${apiBaseUrl}/auth${path}`,
    method,
    data,
    header: {
      'Content-Type': 'application/json',
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    timeout: 15000,
  })
  return { statusCode: response.statusCode, data: response.data }
})

export const accountSession = new AccountStore(authApi, {
  read: () => Taro.getStorageSync(tokenKey),
  write: (token) => Taro.setStorageSync(tokenKey, token),
  remove: () => Taro.removeStorageSync(tokenKey),
}, (reason) => {
  journal.clear()
  clearPhotoCache()
  if (reason === 'expired') {
    // Destroy cached private drafts on expiry, not only on explicit logout.
    void Taro.reLaunch({ url: '/pages/me/index' }).catch(() => {
      void Taro.showToast({ title: '会话失效，请关闭并重新打开应用', icon: 'none' })
    })
  }
})

setUnauthorizedHandler(() => {
  try { accountSession.signOut() } catch { /* AccountStore exposes storage errors. */ }
  void Taro.reLaunch({ url: '/pages/me/index' })
})
accountSession.subscribe(() => {
  const state = accountSession.getSnapshot()
  if (state.status === 'authenticated' && state.user) void journal.connect(state.user.id)
  else if (state.status === 'guest' || state.status === 'error') {
    journal.clear()
    clearPhotoCache()
  }
})

export async function wechatToken(): Promise<string> {
  if (process.env.TARO_ENV !== 'weapp') throw new AuthError(0, '请在微信小程序中使用微信登录。')
  let code: string
  try {
    const response = await Taro.login({ timeout: 10000 })
    code = response.code
  } catch {
    throw new AuthError(0, '未能取得微信登录凭证，请检查网络并重试。')
  }
  // A fresh one-use code on every explicit retry; never cache or log it.
  return authApi.wechat(code || '')
}
