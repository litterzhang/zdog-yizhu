import { Button, Image, Text, View } from '@tarojs/components'
import Taro from '@tarojs/taro'
import { brand } from '../../config/brand'
import { dialogCopy } from '../../config/dialogs'
import { ActionButton } from '../../components/ActionButton'
import { showPhotoCredits } from '../../components/PhotoCredits'
import { useAccount } from '../../hooks/useAccount'
import { useDialog } from '../../hooks/useDialog'
import { useJournal } from '../../hooks/useJournal'
import { accountSession } from '../../services/session'
import { hasVerifiedAccount } from '../../services/account'
import { journal } from '../../services/journal'
import { clearPhotoCache } from '../../services/transport'
import { JournalStatus } from '../../components/JournalStatus'
import { goTab, goTo } from '../../services/navigation'
import './index.scss'

function MenuRow({ title, subtitle, onClick }: { title: string; subtitle?: string; onClick: () => void }) {
  return (
    <Button className='me__menu-row' hoverClass='button--pressed' onClick={onClick}>
      <View className='grow'><Text className='me__menu-title'>{title}</Text>{subtitle && <Text className='block me__menu-subtitle'>{subtitle}</Text>}</View>
      <Text className='me__chevron'>›</Text>
    </Button>
  )
}

export default function MePage() {
  const { showDialog, dialog } = useDialog(true)
  const session = useAccount()
  const account = hasVerifiedAccount(session) ? session.user : null
  const snapshot = useJournal()

  async function clearData(signOut = false) {
    try {
      const response = await showDialog(signOut ? dialogCopy.signOut : dialogCopy.clearData)
      if (!response.confirm) return
      if (signOut) accountSession.signOut()
      else { await journal.deleteAll(); clearPhotoCache() }
      // Destroy cached tab pages as well, so drafts/photos cannot leak after logout.
      await Taro.reLaunch({ url: '/pages/me/index' })
    } catch {
      void Taro.showToast({ title: '操作未完成，请重试', icon: 'none' })
    }
  }

  return (
    <>{dialog}
    <View className='page tab-page me'>
      <View className='me__account'>
        <View className='row me__identity'>
          <View className='me__avatar'><Image src='/assets/line/sprig.png' mode='aspectFit' className='me__avatar-art' ariaLabel='' /></View>
          <View className='grow'>
            <Text className='me__account-name'>{account?.nickname ?? '你好，草木观察者'}</Text>
            <Text className='block me__account-description'>{account ? '已登录 · zdog-auth' : session.status === 'loading' ? '正在验证登录状态…' : '游客模式 · 先从身边的一株开始'}</Text>
          </View>
        </View>
        {!account && <ActionButton disabled={session.status === 'loading'} onClick={() => goTo('/pages/login/index')}>登录我的手账</ActionButton>}
        {session.error && <View className='notice notice--warning'>{session.error}</View>}
        {session.status === 'error' && <Button className='text-button' onClick={() => { void accountSession.restore() }}>重试验证会话</Button>}
        <Text className='me__account-note'>照片与手账私有保存在一株服务器。退出不会删除记录，登录后可以找回。</Text>
      </View>
      {account && <JournalStatus />}
      {account && <View className='me__counts'>
        <Button className='me__count' onClick={() => goTab('footprints')}>
          <Text className='me__count-value'>{snapshot.recognitions.length}</Text>
          <Text className='me__count-label'>我的足迹</Text>
        </Button>
        <View className='me__count-divider' />
        <Button className='me__count' onClick={() => goTab('journal')}>
          <Text className='me__count-value'>{snapshot.plants.length}</Text>
          <Text className='me__count-label'>记住的植物</Text>
        </Button>
      </View>}
      <View className='me__menu'>
        <MenuRow title='数据与隐私' subtitle='我的记录，留给自己' onClick={() => {
          void showDialog(dialogCopy.privacy)
        }}
        />
        <MenuRow title='照片与插画来源' onClick={() => { void showPhotoCredits(showDialog) }} />
        <MenuRow title={`关于${brand.name}`} onClick={() => {
          void showDialog(dialogCopy.about)
        }}
        />
        {account && <MenuRow title='删除我的全部数据' subtitle='删除服务器记录和照片，不删除手机相册原图' onClick={() => { void clearData() }} />}
      </View>
      {(account || session.status === 'error') && <Button className='text-button text-button--center me__logout' onClick={() => { void clearData(true) }}>退出登录 / 清除本机凭据</Button>}
      <View className='me__footer'><Text>每一株，都值得被认真记住。</Text><Text className='block'>{brand.name} · 工作名称</Text></View>
    </View></>
  )
}
