import { Image, Text, View } from '@tarojs/components'
import { ActionButton } from './ActionButton'

interface Props {
  title: string
  description: string
  action?: string
  onAction?: () => void
}

export function StatePanel({ title, description, action, onAction }: Props) {
  return (
    <View className='state'>
      <Image className='state__art' src='/assets/line/sprig.png' mode='aspectFit' ariaLabel='' />
      <Text className='state__title'>{title}</Text>
      <Text className='state__body'>{description}</Text>
      {action && onAction && <ActionButton variant='secondary' onClick={onAction}>{action}</ActionButton>}
    </View>
  )
}
