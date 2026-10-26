import { useState } from 'react'
import { Image, Text, View } from '@tarojs/components'
import type { Photo } from '../types/domain'
import { usePhoto } from '../hooks/usePhoto'

// Keep the real image separate from placeholders. Reusing a mounted Taro H5
// Image while changing lazyLoad from false to true never installs its observer.
export function PhotoThumb({ photo, className = '' }: { photo: Photo; className?: string }) {
  const [failedUri, setFailedUri] = useState('')
  const [loadedUri, setLoadedUri] = useState('')
  const { uri, error, retry } = usePhoto(photo)
  const failed = Boolean(error) || Boolean(uri && failedUri === uri)
  const loading = !uri || loadedUri !== uri
  return (
    <View className={`photo-thumb ${className}`} onClick={(event) => {
      if (failed) {
        event.stopPropagation()
        setFailedUri('')
        setLoadedUri('')
        retry()
      }
    }}
    >
      {uri && !failed && <Image
        key={`photo-${photo.id}-${uri}`}
        className='photo-thumb__image'
        src={uri}
        mode='aspectFill'
        lazyLoad
        onLoad={() => setLoadedUri(uri)}
        onError={() => setFailedUri(uri)}
      />}
      {(loading || failed) && <View className='photo-thumb__placeholder'>
        {failed && <Image key='error-art' className='photo-thumb__fallback' src='/assets/line/sprig.png' mode='aspectFit' ariaLabel='' />}
        <Text className='photo-thumb__status'>{failed ? '照片未加载\n轻点重试' : '照片加载中…'}</Text>
      </View>}
    </View>
  )
}
