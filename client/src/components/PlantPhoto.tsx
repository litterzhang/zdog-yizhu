import { useState } from 'react'
import { Image, Text, View } from '@tarojs/components'
import Taro from '@tarojs/taro'
import type { Photo } from '../types/domain'
import { usePhoto } from '../hooks/usePhoto'

export function PlantPhoto({ photo, height = 236 }: { photo: Photo; height?: number }) {
  const { uri, error, retry } = usePhoto(photo)
  const [loadedUri, setLoadedUri] = useState('')
  const [failedUri, setFailedUri] = useState('')
  const [attempt, setAttempt] = useState(0)
  const failed = Boolean(error) || Boolean(uri && failedUri === uri)

  async function preview() {
    if (failed) {
      setFailedUri('')
      setAttempt((value) => value + 1)
      retry()
      return
    }
    try {
      if (!uri) return
      await Taro.previewImage({ current: uri, urls: [uri] })
    } catch {
      await Taro.showToast({ title: '预览未打开，请重试', icon: 'none' })
    }
  }

  return (
    <View
      className='photo'
      style={{ height: Taro.pxTransform(height) }}
      onClick={() => { void preview() }}
      role='button'
      ariaLabel={failed ? '重新加载照片' : '查看完整照片'}
    >
      {(!uri || loadedUri !== uri || failed) && (
        <View className='photo__placeholder'>{failed ? '照片未加载 · 轻点重试' : '照片加载中…'}</View>
      )}
      {!failed && uri && <Image
        key={`${photo.id}-${attempt}`}
        src={uri}
        className='photo__image'
        mode='aspectFill'
        onLoad={() => setLoadedUri(uri)}
        onError={() => setFailedUri(uri)}
      />}
      {!failed && <Text className='photo__hint'>{photo.origin === 'demo' ? '示例照片 · ' : ''}查看全图</Text>}
    </View>
  )
}
