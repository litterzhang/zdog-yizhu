import { Text, View } from '@tarojs/components'
import { useAccount } from '../hooks/useAccount'
import { useJournal } from '../hooks/useJournal'
import { journal } from '../services/journal'
import { hasVerifiedAccount } from '../services/account'
import { goTo } from '../services/navigation'
import { StatePanel } from './StatePanel'

export function JournalStatus() {
  const session = useAccount()
  const snapshot = useJournal()
  if (session.status === 'loading') return <Text className='block small muted'>正在验证登录状态…</Text>
  if (!hasVerifiedAccount(session)) return <StatePanel title='登录后，找回你的记录' description='照片和手账默认私有，仅你可以查看。' action='登录' onAction={() => goTo('/pages/login/index')} />
  if (snapshot.error) return <StatePanel title='记录暂未加载' description={snapshot.error} action='重试加载' onAction={() => { void journal.refresh() }} />
  return snapshot.loading ? <View className='small muted'>正在同步你的记录…</View> : null
}
