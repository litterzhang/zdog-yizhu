import { useRef, useState } from 'react'
import { Button, Checkbox, CheckboxGroup, Image, Input, Label, Text, View } from '@tarojs/components'
import Taro, { useDidHide, useDidShow, useUnload } from '@tarojs/taro'
import { ActionButton } from '../../components/ActionButton'
import { useAccount } from '../../hooks/useAccount'
import { useDialog } from '../../hooks/useDialog'
import { dialogCopy } from '../../config/dialogs'
import { accountSession, authApi, wechatToken } from '../../services/session'
import { readableError } from '../../services/media'
import { goHome } from '../../services/navigation'
import './index.scss'

export default function LoginPage() {
  const { showDialog, dialog } = useDialog()
  const [consented, setConsented] = useState(false)
  const [error, setError] = useState('')
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const session = useAccount()
  const active = useRef(true)
  const isWeapp = process.env.TARO_ENV === 'weapp'
  const busy = session.status === 'loading'
  useDidShow(() => { active.current = true })
  useDidHide(() => { active.current = false; setPassword('') })
  useUnload(() => { active.current = false })

  async function signIn() {
    setError('')
    const submittedPassword = password
    setPassword('')
    try {
      await accountSession.signIn(consented, isWeapp
        ? wechatToken
        : () => authApi.password(username, submittedPassword))
      // Remove all old page drafts after switching from guest to verified account.
      if (active.current) await Taro.reLaunch({ url: '/pages/me/index' })
    } catch (err) {
      if (active.current) setError(readableError(err))
    }
  }

  function showInfo(privacy: boolean) {
    void showDialog(privacy ? dialogCopy.loginPrivacy : dialogCopy.usage)
  }

  return (
    <>{dialog}
    <View className='page login'>
      <Image src='/assets/line/branch.png' mode='aspectFit' className='login__art' ariaLabel='' />
      <Text className='title login__title'>把相遇，慢慢记下来</Text>
      <Text className='login__subtitle'>一份手账，留给身边的草木。</Text>
      <View className='login__benefits'>
        <View className='login__benefit'><Image src='/assets/tabs/footprints-selected.png' className='icon' ariaLabel='' /><View><Text className='login__benefit-title'>找回每一次发现</Text><Text className='block small muted'>足迹里，是你认识过的植物。</Text></View></View>
        <View className='login__benefit'><Image src='/assets/tabs/journal-selected.png' className='icon' ariaLabel='' /><View><Text className='login__benefit-title'>记住特别的这一株</Text><Text className='block small muted'>用株记，继续记录它的变化。</Text></View></View>
      </View>
      {!isWeapp && (
        <View className='login__credentials'>
          <Text className='block small muted'>使用已有的 zdog-auth 账号登录。微信登录请在小程序内使用。</Text>
          <Label className='login__field'>
            <Text>账号</Text>
            <Input className='login__input' value={username} maxlength={128} disabled={busy} placeholder='请输入账号' onInput={(event) => setUsername(event.detail.value)} />
          </Label>
          <Label className='login__field'>
            <Text>密码</Text>
            <Input className='login__input' value={password} password maxlength={256} disabled={busy} placeholder='请输入密码' onInput={(event) => setPassword(event.detail.value)} />
          </Label>
        </View>
      )}
      <CheckboxGroup onChange={(event) => { setConsented(event.detail.value.includes('auth')); setError('') }}>
        <Label className='login__consent'>
          <Checkbox value='auth' checked={consented} disabled={busy} color='#496B50' />
          <Text>我已阅读并同意登录与隐私说明。</Text>
        </Label>
      </CheckboxGroup>
      <View className='login__policy-links'>
        <Button className='text-button' onClick={() => showInfo(false)}>{dialogCopy.usage.title}</Button>
        <Text className='muted small'>与</Text>
        <Button className='text-button' onClick={() => showInfo(true)}>隐私说明</Button>
      </View>
      {(error || session.error) && <View className='notice notice--warning'>{error || session.error}</View>}
      <ActionButton disabled={!consented || (!isWeapp && (!username.trim() || !password))} loading={busy} onClick={() => { void signIn() }}>
        {busy ? '正在验证身份…' : isWeapp ? '微信登录' : '账号密码登录'}
      </ActionButton>
      <Button className='text-button text-button--center login__guest' onClick={goHome}>先以游客身份逛逛</Button>
      <Text className='block subtle-footer'>{isWeapp ? '不需要手机号、头像或微信密码。' : '密码仅用于本次认证，不保存在本机。'}</Text>
    </View></>
  )
}
