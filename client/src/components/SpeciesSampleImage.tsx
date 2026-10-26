import { useState } from 'react'
import { Button, Image, Text, View } from '@tarojs/components'
import './SpeciesSampleImage.scss'

export function SpeciesSampleImage({ url, name }: { url: string; name: string }) {
  const [status, setStatus] = useState<'loading' | 'loaded' | 'failed'>('loading')
  const [attempt, setAttempt] = useState(0)

  return <View className='species-sample'>
    <Text className='species-sample__label'>植物样图</Text>
    <View className='species-sample__frame'>
      {status !== 'failed' && <Image
        key={attempt}
        className='species-sample__image'
        src={url}
        mode='aspectFit'
        ariaLabel={`${name}的植物样图`}
        onLoad={() => setStatus('loaded')}
        onError={() => setStatus('failed')}
      />}
      {status === 'loading' && <View className='species-sample__status'><Text>图片加载中…</Text></View>}
      {status === 'failed' && <View className='species-sample__status'>
        <Text>图片暂未加载</Text>
        <Button className='text-button' onClick={() => { setAttempt((value) => value + 1); setStatus('loading') }}>重新加载</Button>
      </View>}
    </View>
  </View>
}
