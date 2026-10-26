import Taro from '@tarojs/taro'
import { apiBaseUrl } from '../config/api'
import { AuthError, validToken } from './auth-api'
import type { Photo } from '../types/domain'
import { downloadOptions, parseResponse, transferFailure, uploadOptions } from './file-transfer'

export const tokenKey = 'yizhu.auth.token.v1'
let unauthorized = () => {}
export function setUnauthorizedHandler(handler: () => void) { unauthorized = handler }
export function authToken(): string {
  const token: unknown = Taro.getStorageSync(tokenKey)
  if (!validToken(token)) throw new AuthError(401, '请先登录。')
  return token
}
function checkSession(token: string, status: number) {
  if (Taro.getStorageSync(tokenKey) !== token) throw new Error('账号已切换，请重新操作。')
  if (status === 401) { unauthorized(); throw new AuthError(401, '登录已过期，请重新登录。') }
}
export async function apiRequest(path: string, data?: Record<string, unknown>): Promise<unknown> {
  const token = authToken()
  let response: { statusCode: number; data: unknown }
  try {
    response = await Taro.request({
      url: `${apiBaseUrl}${path}`, method: data === undefined ? 'GET' : 'POST', data,
      header: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' }, timeout: 20000,
    })
  } catch { throw new Error('无法连接服务，请检查网络后重试。') }
  checkSession(token, response.statusCode)
  return parseResponse(response.statusCode, response.data)
}
export async function uploadPhoto(photo: Photo): Promise<Photo> {
  if (photo.origin === 'remote') return photo
  if (photo.origin !== 'local') throw new Error('请选择自己的照片进行真实识别。')
  const token = authToken()
  let response: Taro.uploadFile.SuccessCallbackResult
  try {
    response = await Taro.uploadFile(uploadOptions(apiBaseUrl, photo, token))
  } catch (error) { throw transferFailure('upload', error, process.env.TARO_ENV === 'h5') }
  checkSession(token, response.statusCode)
  const value = parseResponse(response.statusCode, response.data) as Photo
  if (!value || typeof value.id !== 'string' || value.origin !== 'remote') throw new Error('照片响应不完整，请重试。')
  return value
}

let photoEpoch = 0
const photoCache = new Map<string, Promise<string>>()
const downloadedFiles = new Set<string>()
function removeTemporaryPhoto(path: string) {
  if (process.env.TARO_ENV === 'weapp' && path) {
    try { Taro.getFileSystemManager().unlink({ filePath: path, fail: () => {} }) } catch { /* Best effort. */ }
  }
}
export function clearPhotoCache() {
  photoEpoch++
  photoCache.clear()
  downloadedFiles.forEach(removeTemporaryPhoto)
  downloadedFiles.clear()
}
export function privatePhoto(photo: Photo): Promise<string> {
  if (photo.origin !== 'remote') return Promise.resolve(photo.uri)
  const existing = photoCache.get(photo.id)
  if (existing) return existing
  const token = authToken()
  const epoch = photoEpoch
  const promise = Taro.downloadFile(downloadOptions(apiBaseUrl, photo.id, token)).then((response) => {
    try {
      checkSession(token, response.statusCode)
      if (epoch !== photoEpoch) throw new Error('账号状态已变化。')
      if (response.statusCode !== 200) throw new Error('照片暂不可用，请重试。')
      downloadedFiles.add(response.tempFilePath)
      return response.tempFilePath
    } catch (error) {
      removeTemporaryPhoto(response.tempFilePath)
      throw error
    }
  }).catch((error) => {
    if (epoch === photoEpoch) photoCache.delete(photo.id)
    if (error instanceof Error) throw error
    throw transferFailure('download', error, process.env.TARO_ENV === 'h5')
  })
  if (photoCache.size >= 60) photoCache.delete(photoCache.keys().next().value!)
  photoCache.set(photo.id, promise)
  return promise
}
export function retryPrivatePhoto(id: string) { photoCache.delete(id) }
