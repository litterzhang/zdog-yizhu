import { Text, View } from '@tarojs/components'
import { brand } from '../config/brand'

export function DemoNote() {
  return (
    <View className='demo-note'>
      <Text className='block'>{brand.demoLabel}</Text>
      <Text className='block'>{brand.sessionNotice}</Text>
    </View>
  )
}
