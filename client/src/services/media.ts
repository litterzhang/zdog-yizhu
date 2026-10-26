import Taro from '@tarojs/taro'
import type { Photo } from '../types/domain'
import { createLocalId } from './ids'
import type { ShowDialog } from './dialog'
import { dialogCopy } from '../config/dialogs'

export class MediaError extends Error {
  constructor(message: string, readonly permissionDenied = false) {
    super(message)
  }
}

export async function choosePhoto(source: 'camera' | 'album'): Promise<Photo | null> {
  try {
    // Browsers control whether a camera capture hint is honored.
    if (process.env.TARO_ENV === 'h5') {
      const result = await Taro.chooseImage({
        count: 1,
        sourceType: [source],
        sizeType: ['compressed'],
      })
      const file = Array.isArray(result.tempFiles) ? result.tempFiles[0] : undefined
      if (!file) throw new MediaError('没有读到照片，请重新选择。')
      if (file.size > 900 * 1024) throw new MediaError('网页版请先选择小于 900 KiB 的照片；小程序会自动尝试压缩。')
      return { id: createLocalId('photo'), uri: file.path, origin: 'local' }
    }
    const result = await Taro.chooseMedia({
      count: 1,
      mediaType: ['image'],
      sourceType: [source],
      sizeType: ['compressed'],
    })
    const file = result.tempFiles[0]
    if (!file) throw new MediaError('没有读到照片，请重新选择。')
    if (file.size > 10 * 1024 * 1024) throw new MediaError('照片超过 10 MB，请选择较小的照片后重试。')
    let path = file.tempFilePath
    if (file.size > 850 * 1024) {
      const compressed = await Taro.compressImage({ src: path, quality: 65, compressedWidth: 1400 })
      path = compressed.tempFilePath
      const info = await Taro.getFileInfo({ filePath: path })
      if (!('size' in info)) throw new MediaError('无法确认压缩后的照片大小，请重新选择。')
      if (info.size > 900 * 1024) throw new MediaError('压缩后仍超过 900 KiB，请裁剪或选择较小的照片。')
    }
    return { id: createLocalId('photo'), uri: path, origin: 'local' }
  } catch (error) {
    if (error instanceof MediaError) throw error
    const message = (error as { errMsg?: string }).errMsg ?? ''
    if (/cancel/i.test(message)) return null
    if (/auth deny|auth denied|authorize|permission|privacy/i.test(message)) {
      throw new MediaError('尚未获得相机或相册权限，请检查权限设置后重试。', true)
    }
    throw new MediaError('暂时无法打开照片，请重试或重新选择照片。')
  }
}

export async function openPermissionSettings(showDialog: ShowDialog) {
  if (process.env.TARO_ENV === 'h5') {
    await showDialog(dialogCopy.browserPermissions)
    return
  }
  try {
    await Taro.openSetting()
  } catch {
    await showDialog(dialogCopy.wechatPermissions)
  }
}

export function readableError(error: unknown): string {
  return error instanceof Error ? error.message : '操作未完成，请重试。'
}
