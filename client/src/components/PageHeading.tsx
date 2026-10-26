import { Text, View } from '@tarojs/components'

export function PageHeading({ title, subtitle, label }: { title: string; subtitle: string; label: string }) {
  return (
    <View className='page-heading'>
      <Text className='eyebrow'>{label}</Text>
      <Text className='title page-heading__title'>{title}</Text>
      <Text className='block muted page-heading__subtitle'>{subtitle}</Text>
    </View>
  )
}
