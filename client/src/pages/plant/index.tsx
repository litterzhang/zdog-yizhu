import { useRef, useState } from 'react'
import { Button, Input, Text, Textarea, View } from '@tarojs/components'
import Taro, { useRouter } from '@tarojs/taro'
import { ActionButton } from '../../components/ActionButton'
import { JournalStatus } from '../../components/JournalStatus'
import { PlantPhoto } from '../../components/PlantPhoto'
import { StatePanel } from '../../components/StatePanel'
import { SpeciesSearch } from '../../components/SpeciesSearch'
import { brand } from '../../config/brand'
import { dialogCopy } from '../../config/dialogs'
import { useJournal } from '../../hooks/useJournal'
import { useDialog } from '../../hooks/useDialog'
import { journal } from '../../services/journal'
import { findSpecies } from '../../services/catalog'
import { choosePhoto, MediaError, openPermissionSettings, readableError } from '../../services/media'
import { uploadPhoto } from '../../services/transport'
import { goTab, goTo } from '../../services/navigation'
import type { Photo } from '../../types/domain'
import './index.scss'

export default function PlantPage() {
  const { showDialog, dialog } = useDialog()
  const { params } = useRouter()
  const snapshot = useJournal()
  const plant = snapshot.plants.find((item) => item.id === params.id)
  const observations = snapshot.observations.filter((item) => item.plantId === plant?.id)
    .sort((left, right) => Date.parse(right.observedAt) - Date.parse(left.observedAt))
  const botanical = findSpecies(plant?.speciesId)
  const [adding, setAdding] = useState(false)
  const [photo, setPhoto] = useState<Photo | null>(null)
  const [note, setNote] = useState('')
  const [error, setError] = useState('')
  const [denied, setDenied] = useState(false)
  const [choosing, setChoosing] = useState(false)
  const [busy, setBusy] = useState(false)
  const submitting = useRef(false)
  const [editing, setEditing] = useState(false)
  const [nickname, setNickname] = useState('')
  const [location, setLocation] = useState('')
  const [story, setStory] = useState('')
  const [description, setDescription] = useState('')
  const [speciesId, setSpeciesId] = useState<string | null>(null)
  const [showSpeciesSearch, setShowSpeciesSearch] = useState(false)
  const [editingObservation, setEditingObservation] = useState('')
  const [observationNote, setObservationNote] = useState('')

  async function perform(action: () => Promise<unknown>) {
    if (submitting.current) return
    submitting.current = true
    setBusy(true)
    setError('')
    try { await action() } catch (err) { setError(readableError(err)) }
    finally { submitting.current = false; setBusy(false) }
  }
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
  async function saveObservation() {
    if (!plant || !photo) return
    await perform(async () => {
      const consent = await showDialog(dialogCopy.saveObservation)
      if (!consent.confirm) return
      const remote = await uploadPhoto(photo)
      await journal.appendObservation(plant.id, remote.id, note, `${plant.id}:${photo.id}`)
      setAdding(false)
      setPhoto(null)
      setNote('')
      void Taro.showToast({ title: '观察已保存', icon: 'success' })
    })
  }

  if (!plant) return <>{dialog}<View className='page'><JournalStatus /><StatePanel title={snapshot.loading ? '正在找回这一株…' : '这株档案暂不可用'} description='请确认登录状态并重试加载，或回到株记查看其他植物。' action='重新加载' onAction={() => { void journal.refresh() }} /></View></>

  return (
    <>{dialog}
    <View className='page plant'>
      <View className='row between plant__heading'><Text className='eyebrow'>株记 / 个体档案</Text><Text className='badge'>私有档案</Text></View>
      <JournalStatus />
      <PlantPhoto photo={plant.cover} height={218} />
      <View className='plant__identity'>
        <Text className='title'>{plant.nickname}</Text>
        <Text className='block muted'>{botanical ? `${botanical.commonName} · ${botanical.scientificName}` : '物种待确认'}</Text>
        {botanical && botanical.isVisible !== false && <Button className='text-button small' onClick={() => goTo(`/pages/catalog/index?id=${encodeURIComponent(botanical.id)}`)}>查看物种分类与来源</Button>}
        <Text className='block small muted'>{botanical ? '物种由用户选择，不代表专业鉴定' : '还不知道名字，也不妨碍记录它'}</Text>
      </View>
      {plant.locationDescription && <Text className='plant__location block'>遇见于 · {plant.locationDescription}</Text>}
      <View className='plant__story'><Text className='eyebrow'>我和它的故事</Text><Text className='block'>{plant.story || '故事还没开始写。先记住它的样子。'}</Text><Text className='block small muted'>用户内容 · 不作为物种知识或树龄证明</Text></View>
      <View className='section plant__description'>
        <Text className='section-title'>我的描述</Text>
        <Text className='block small muted'>只描述这一株 · 私人记录，不是物种鉴定结论</Text>
        <Text className='block plant__description-text'>{plant.description || '还没有描述，可在“编辑植物档案”中记录它的特征。'}</Text>
      </View>
      <ActionButton icon='plus-white' disabled={busy} onClick={() => { setAdding(!adding); setError('') }}>记录新变化</ActionButton>
      {error && <View className='notice notice--warning'>{error}</View>}
      {adding && <View className='form'>
        <Text className='section-title'>今天的它，有什么不同？</Text>
        <Text className='block small muted'>追加到「{plant.nickname}」，不会创建另一个植物个体。</Text>
        <View className='stack section'>
          <ActionButton variant='secondary' loading={choosing} disabled={busy} onClick={() => { void pick('camera') }}>拍一张照片</ActionButton>
          <ActionButton variant='secondary' disabled={choosing || busy} onClick={() => { void pick('album') }}>从相册选择</ActionButton>
          {photo && <PlantPhoto photo={photo} height={184} />}
        </View>
        <View className='field'><Text className='field__label'>观察笔记 · 可选，最多 500 字</Text><Textarea className='field__textarea' placeholder='一片新叶、一阵花香，或是一次路过。' maxlength={500} value={note} onInput={(event) => setNote(event.detail.value)} /></View>
        <Text className='block small muted'>{brand.uploadNotice}</Text>
        {denied && <Button className='text-button' onClick={() => { void openPermissionSettings(showDialog) }}>检查权限设置</Button>}
        <ActionButton disabled={!photo || choosing} loading={busy} onClick={() => { void saveObservation() }}>保存这次观察</ActionButton>
        <Button className='text-button' disabled={busy || undefined} onClick={() => { setAdding(false); setPhoto(null); setNote('') }}>取消</Button>
      </View>}
      <View className='section'>
        <View className='row between'><Text className='section-title'>慢慢长成的故事</Text><Text className='small muted'>{observations.length} 次观察</Text></View>
        <View className='plant__timeline'>
          {observations.length ? observations.map((observation) => <View className='plant__entry' key={observation.id}>
            <View className='plant__node' />
            <Text className='plant__date'>{new Date(observation.observedAt).toLocaleDateString('zh-CN', { year: 'numeric', month: 'long', day: 'numeric' })}</Text>
            <Text className='plant__note block'>{observation.note || '今天，留下了一张照片。'}</Text>
            <PlantPhoto photo={observation.photo} height={150} />
            {editingObservation === observation.id && <View className='field'>
              <Textarea className='field__textarea' maxlength={500} value={observationNote} onInput={(event) => setObservationNote(event.detail.value)} />
              <ActionButton loading={busy} onClick={() => { void perform(async () => { await journal.editObservation(observation.id, observationNote); setEditingObservation('') }) }}>保存笔记</ActionButton>
              <Button className='text-button' onClick={() => setEditingObservation('')}>取消编辑</Button>
            </View>}
            <View className='row between'>
              <Button className='text-button small' disabled={busy || undefined} onClick={() => { setEditingObservation(observation.id); setObservationNote(observation.note) }}>编辑笔记</Button>
              <Button className='text-button small' disabled={busy || undefined} onClick={() => { void perform(async () => {
                const answer = await showDialog(dialogCopy.removeObservation)
                if (answer.confirm) await journal.remove('observations', observation.id)
              }) }}
              >删除观察</Button>
            </View>
          </View>) : <StatePanel title='等待下一次相遇' description='给这一株拍一张照片，写下今天看到的变化。' action='记录新变化' onAction={() => setAdding(true)} />}
        </View>
      </View>
      {editing && <View className='form'>
        <Text className='section-title'>编辑这一株</Text>
        <View className='field'><Text className='field__label'>昵称</Text><Input className='field__input' value={nickname} maxlength={40} onInput={(event) => setNickname(event.detail.value)} /></View>
        <View className='field'><Text className='field__label'>地点描述</Text><Input className='field__input' value={location} maxlength={80} onInput={(event) => setLocation(event.detail.value)} /></View>
        <View className='field'><Text className='field__label'>我的故事</Text><Textarea className='field__textarea' value={story} maxlength={500} onInput={(event) => setStory(event.detail.value)} /></View>
        <View className='field'><Text className='field__label'>我的描述 · 最多 1000 字</Text><Textarea className='field__textarea plant__description-input' value={description} maxlength={1000} placeholder='记录这一株的外观与特征，不会公开到物种名录。' onInput={(event) => setDescription(event.detail.value)} /></View>
        <Button className='text-button' disabled={busy || undefined} onClick={() => setShowSpeciesSearch(!showSpeciesSearch)}>物种：{findSpecies(speciesId)?.commonName ?? '待确认'} · 从名录选择</Button>
        {showSpeciesSearch && <SpeciesSearch disabled={busy} onSelect={(species) => { setSpeciesId(species.id); setShowSpeciesSearch(false) }} />}
        <Button className='text-button small' disabled={busy || undefined} onClick={() => setSpeciesId(null)}>保留待确认物种</Button>
        <Text className='block small muted'>修改昵称或物种不改变个体 ID，也不改写原始识别结果。</Text>
        <ActionButton disabled={!nickname.trim()} loading={busy} onClick={() => { void perform(async () => { await journal.editPlant(plant.id, { nickname, location, story, speciesId, description }); setEditing(false) }) }}>保存档案</ActionButton>
        <Button className='text-button' onClick={() => setEditing(false)}>取消编辑</Button>
      </View>}
      <View className='toolbar'>
        <Button className='text-button' disabled={busy || undefined} onClick={() => { setEditing(!editing); setNickname(plant.nickname); setLocation(plant.locationDescription); setStory(plant.story); setDescription(plant.description ?? ''); setSpeciesId(plant.speciesId) }}>编辑植物档案</Button>
        <Button className='text-button' disabled={busy || undefined} onClick={() => { void perform(async () => {
          const answer = await showDialog(dialogCopy.removePlant(plant.nickname))
          if (answer.confirm) { await journal.remove('plants', plant.id); goTab('journal') }
        }) }}
        >删除植物档案</Button>
        <Text className='block small muted'>个体 ID：{plant.id}</Text>
        <Text className='block small muted'>昵称与物种不作为个体身份。照片不会自动匹配同一株。</Text>
      </View>
    </View></>
  )
}
