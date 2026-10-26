import type Taro from '@tarojs/taro'
import type { Photo } from '../types/domain'

// Taro H5 file APIs default to withCredentials=true, unlike our bearer-only API.
// Explicitly disable cookies without removing the Authorization header.
export function uploadOptions(base: string, photo: Photo, token: string): Taro.uploadFile.Option {
  return {
    url: `${base}/photos`, filePath: photo.uri, name: 'file',
    withCredentials: false,
    header: { Authorization: `Bearer ${token}` },
    formData: { requestId: photo.id }, timeout: 30000,
  }
}

export function downloadOptions(base: string, photoId: string, token: string): Taro.downloadFile.Option {
  return {
    url: `${base}/photos/content`,
    withCredentials: false,
    header: { Authorization: `Bearer ${token}`, 'X-Photo-Id': photoId },
    timeout: 20000,
  }
}

export function parseResponse(status: number, data: unknown): unknown {
  let value = data
  if (typeof value === 'string') {
    try { value = JSON.parse(value) } catch { value = null }
  }
  if (status < 200 || status >= 300) {
    // An upstream proxy may send HTML; do not lose its actionable HTTP status.
    if (status === 413) throw new Error('照片过大，请压缩到 900 KiB 以下再试。')
    if (status === 401) throw new Error('登录已过期，请重新登录。')
    const message = value && typeof value === 'object' && 'message' in value ? value.message : null
    if (typeof message === 'string' && message.length <= 200) throw new Error(message)
    if (status >= 500) throw new Error('服务暂不可用，请稍后重试。')
    throw new Error('请求未完成，请稍后重试。')
  }
  if (value === null || typeof value !== 'object') throw new Error('服务响应异常，请稍后重试。')
  return value
}

export function transferFailure(action: 'upload' | 'download', error: unknown, h5: boolean): Error {
  const message = error && typeof error === 'object' && 'errMsg' in error ? error.errMsg : ''
  const detail = typeof message === 'string' ? message : ''
  const label = action === 'upload' ? '照片上传' : '照片加载'
  if (/timeout/i.test(detail)) return new Error(`${label}超时，请检查网络后重试。`)
  if (/abort|cancel/i.test(detail)) return new Error(`${label}已取消，可以重试。`)
  if (/ENOENT|not found|not exist|read.*file|blob/i.test(detail)) {
    return new Error('照片临时文件已失效，请重新选择照片。')
  }
  return new Error(h5
    ? `${label}失败，请刷新页面后重试；若仍失败，可能是网络或跨域配置问题。`
    : `${label}失败，请检查网络和${action === 'upload' ? '上传' : '下载'}合法域名后重试。`)
}
