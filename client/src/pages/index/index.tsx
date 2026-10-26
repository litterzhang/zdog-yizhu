import { useRef, useState } from 'react'
import { Button, Image, Text, View } from '@tarojs/components'
import { brand } from '../../config/brand'
import { ActionButton } from '../../components/ActionButton'
import { PlantPhoto } from '../../components/PlantPhoto'
import { StatePanel } from '../../components/StatePanel'
import { journal } from '../../services/journal'
import { choosePhoto, MediaError, openPermissionSettings, readableError } from '../../services/media'
import { uploadPhoto } from '../../services/transport'
import { useAccount } from '../../hooks/useAccount'
import { useDialog } from '../../hooks/useDialog'
import { useJournal } from '../../hooks/useJournal'
import { goTo } from '../../services/navigation'
import type { Photo } from '../../types/domain'
import './index.scss'

export default function HomePage() {
  const { showDialog, dialog } = useDialog(true)
  const [photo, setPhoto] = useState<Photo | null>(null)
  const [choosing, setChoosing] = useState(false)
  const [error, setError] = useState('')
  const [denied, setDenied] = useState(false)
  const [uploading, setUploading] = useState(false)
  const submitting = useRef(false)
  const session = useAccount()
  const snapshot = useJournal()

  async function pick(source: 'camera' | 'album') {
    setChoosing(true)
    setError('')
    setDenied(false)
    try {
      const selected = await choosePhoto(source)
      if (selected) setPhoto(selected)
    } catch (err) {
      setError(readableError(err))
      setDenied(err instanceof MediaError && err.permissionDenied)
    } finally { setChoosing(false) }
  }

  async function recognize() {
    if (!photo || submitting.current) return
    if (!session.user) { goTo('/pages/login/index'); return }
    submitting.current = true
    setError('')
    try {
      setUploading(true)
      const remote = await uploadPhoto(photo)
      const record = await journal.startRecognition(remote.id, photo.id)
      goTo(`/pages/result/index?id=${encodeURIComponent(record.id)}`)
    } catch (err) { setError(readableError(err)) }
    finally { submitting.current = false; setUploading(false) }
  }

  return (
    <>{dialog}
    <View className='page tab-page home'>
      <View className='home__hero'>
        <View className='home__illustration'>
          <View className='home__soft-circle' />
          <Image src='/assets/line/branch.png' className='home__drawing' mode='aspectFit' ariaLabel='' />
        </View>
        <Text className='home__annotation'>从一片叶子开始，认识身边的自然</Text>
      </View>
      <View className='stack home__actions'>
        <ActionButton icon='camera-white' loading={choosing} disabled={uploading} onClick={() => { void pick('camera') }}>拍照识别</ActionButton>
        <ActionButton variant='secondary' disabled={choosing || uploading} icon='album' onClick={() => { void pick('album') }}>从相册选择</ActionButton>
      </View>
      <Button className='text-button text-button--center' onClick={() => goTo('/pages/catalog/index')}>浏览植物名录 · 目 / 科 / 属 / 种</Button>
      {error && <View className='stack home__selected'>
        <StatePanel title={denied ? '还没有照片权限' : '操作没有完成'} description={error} action='重新选择照片' onAction={() => { void pick('album') }} />
        {denied && <ActionButton variant='secondary' onClick={() => { void openPermissionSettings(showDialog) }}>检查权限设置</ActionButton>}
      </View>}
      {photo && <View className='home__selected stack'>
        <View className='row between'><Text className='section-title'>这一次遇见</Text><Button className='text-button' disabled={uploading || undefined} onClick={() => setPhoto(null)}>重新选择</Button></View>
        <PlantPhoto photo={photo} height={220} />
        <Text className='small muted'>{brand.recognitionNotice}</Text>
        {snapshot.engine === 'warming' && <Text className='small muted'>模型正在启动，任务会保存到足迹并排队处理。</Text>}
        <ActionButton loading={uploading} onClick={() => { void recognize() }}>{session.user ? '上传并识别' : '登录后识别'}</ActionButton>
      </View>}
      {!photo && <View className='home__tips'>
        <Text className='home__tips-title'>给相遇，留一张清晰的照片</Text>
        <View className='row between home__tips-items'><Text>靠近一点</Text><Text className='home__dot'>·</Text><Text>光线自然</Text><Text className='home__dot'>·</Text><Text>叶片清晰</Text></View>
      </View>}
      <Text className='block subtle-footer'>认识物种，也记住这一株。候选不等于鉴定结论。</Text>
    </View></>
  )
}
