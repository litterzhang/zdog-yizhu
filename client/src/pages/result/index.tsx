import { useRef, useState } from 'react'
import { Button, Input, Text, Textarea, View } from '@tarojs/components'
import Taro, { useRouter } from '@tarojs/taro'
import { ActionButton } from '../../components/ActionButton'
import { JournalStatus } from '../../components/JournalStatus'
import { brand } from '../../config/brand'
import { dialogCopy } from '../../config/dialogs'
import { PlantPhoto } from '../../components/PlantPhoto'
import { StatePanel } from '../../components/StatePanel'
import { SpeciesSearch } from '../../components/SpeciesSearch'
import { SpeciesKnowledge } from '../../components/SpeciesKnowledge'
import { useJournal } from '../../hooks/useJournal'
import { useDialog } from '../../hooks/useDialog'
import { journal } from '../../services/journal'
import { findSpecies } from '../../services/catalog'
import { recognitionCandidates, recognitionPreview, speciesSampleImageUrl, type CandidateSelection } from '../../services/presentation'
import { readableError } from '../../services/media'
import { goHome, goTab, goTo } from '../../services/navigation'
import './index.scss'

export default function ResultPage() {
  const { showDialog, dialog } = useDialog()
  const { params } = useRouter()
  const snapshot = useJournal()
  const record = snapshot.recognitions.find((item) => item.id === params.id)
  const [showCandidates, setShowCandidates] = useState(false)
  const [selection, setSelection] = useState<CandidateSelection | null>(null)
  const [showForm, setShowForm] = useState(false)
  const [showPlants, setShowPlants] = useState(false)
  const [showCatalog, setShowCatalog] = useState(false)
  const [showKnowledge, setShowKnowledge] = useState(false)
  const [nickname, setNickname] = useState('')
  const [location, setLocation] = useState('')
  const [story, setStory] = useState('')
  const [description, setDescription] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const submitting = useRef(false)
  const { candidates, hiddenLeading } = recognitionCandidates(record)
  const confirmed = findSpecies(record?.confirmedSpeciesId)
  const { species: previewSpecies, isConfirmed: previewConfirmed } = recognitionPreview(record, selection)
  const knowledgeSpecies = previewSpecies
  const linkedPlant = snapshot.plants.find((item) => item.id === record?.linkedPlantId)
  const associationConflict = record?.associationConflict === true
  const canAssociate = !record?.linkedPlantId && !associationConflict && record?.status !== 'running'

  async function perform(action: () => Promise<unknown>) {
    if (submitting.current) return
    submitting.current = true
    setBusy(true)
    setError('')
    try { await action() } catch (err) { setError(readableError(err)) }
    finally { submitting.current = false; setBusy(false) }
  }

  async function confirmSpecies(speciesId: string | null) {
    if (!record || associationConflict) return
    await perform(async () => {
      await journal.confirmSpecies(record.id, speciesId)
      setSelection(null)
      setShowCandidates(false)
      setShowCatalog(false)
      setShowKnowledge(false)
      void Taro.showToast({ title: speciesId ? '物种已确认' : '已保留待确认', icon: speciesId ? 'success' : 'none' })
    })
  }

  function selectSpecies(speciesId: string, source: CandidateSelection['source'] = 'candidate') {
    if (!record || submitting.current || associationConflict) return
    setSelection({ recognitionId: record.id, speciesId, source })
    setShowCandidates(false)
    setShowCatalog(false)
    setShowKnowledge(false)
    setError('')
  }

  async function createPlant() {
    if (!record || !canAssociate) return
    await perform(async () => {
      const plant = await journal.createPlant(record.id, { nickname, location, story, description })
      setShowForm(false)
      goTo(`/pages/plant/index?id=${encodeURIComponent(plant.id)}`)
    })
  }

  async function linkPlant(plantId: string, plantName: string) {
    if (!record || !canAssociate) return
    await perform(async () => {
      const response = await showDialog(dialogCopy.linkPlant(plantName))
      if (!response.confirm) return
      await journal.linkRecognition(plantId, record.id)
      goTo(`/pages/plant/index?id=${encodeURIComponent(plantId)}`)
    })
  }

  if (!record) return <>{dialog}<View className='page'><JournalStatus /><StatePanel title={snapshot.loading ? '正在找回记录…' : '这次记录暂不可用'} description='请确认登录状态并重试加载，或回到发现重新选择照片。' action='重新加载' onAction={() => { void journal.refresh() }} /></View></>

  const catalogSpecies = previewSpecies
  const catalogLink = catalogSpecies && catalogSpecies.isVisible !== false
    ? <Button className='text-button' onClick={() => goTo(`/pages/catalog/index?id=${encodeURIComponent(catalogSpecies.id)}`)}>查看分类与来源</Button>
    : null
  const showSpeciesPanel = Boolean(previewSpecies || candidates.length > 0 || record.status === 'succeeded')
  const searchAction = <Button className='text-button small' disabled={(busy || associationConflict) || undefined} onClick={() => setShowCatalog(!showCatalog)}>{showCatalog ? '收起名录搜索' : '搜索其他物种'}</Button>

  return (
    <>{dialog}
    <View className='page result'>
      <PlantPhoto photo={record.photo} height={244} />
      <JournalStatus />
      {error && <View className='notice notice--warning'>{error}</View>}
      {associationConflict && <View className='notice notice--warning'>这条足迹关联了多个株记。请先在株记中整理关联，再确认物种或建档；已有记录保留。</View>}
      {record.status === 'running' ? <View className='result__progress'>
        <StatePanel title={snapshot.engine === 'failed' ? '识别服务暂不可用' : snapshot.engine === 'warming' ? '模型正在准备…' : '正在识别…'} description={snapshot.engine === 'failed' ? '任务已保存，服务恢复后继续处理。可稍后到足迹查看。' : '任务已保存，可以离开，稍后到足迹查看结果。'} action='去足迹看看' onAction={() => goTab('footprints')} />
      </View>
        : record.status === 'failed' ? <StatePanel title='这次没有完成' description={record.failure ?? '识别失败，请重试。'} action='重试识别' onAction={() => { void perform(() => journal.retryRecognition(record.id)) }} />
          : record.status === 'uncertain' || hiddenLeading ? <StatePanel title='暂时还认不准' description='换张照片试试，也可以先建档，物种留待确认。' action='重新选择照片' onAction={goHome} />
            : null}
      {record.status !== 'running' && showSpeciesPanel && <View className='result__species'>
        {(previewSpecies || (record.status === 'succeeded' && !hiddenLeading)) && <>
          <View className='row between'><Text className='title'>{previewSpecies?.commonName ?? '待确认'}</Text><Text className='badge'>{previewConfirmed ? '已确认' : '候选待确认'}</Text></View>
          <Text className='result__latin'>{previewSpecies?.scientificName ?? ''}</Text>
          {catalogLink}
        </>}
        <View className='result__confirmation'>
          {!previewConfirmed && previewSpecies && <ActionButton disabled={associationConflict} loading={busy} onClick={() => { void confirmSpecies(previewSpecies.id) }}>确认是「{previewSpecies.commonName}」</ActionButton>}
          <View className='result__confirmation-actions'>
            {candidates.length > 0 && <Button className='text-button small' disabled={busy || undefined} onClick={() => setShowCandidates(!showCandidates)}>{showCandidates ? '收起其他候选' : '选择其他候选'}</Button>}
            {searchAction}
          </View>
          <Button className='text-button small' disabled={(busy || associationConflict) || undefined} onClick={() => { void confirmSpecies(null) }}>暂不确认物种</Button>
        </View>
        {showCandidates && <View>
          {candidates.map((candidate) => {
            const item = findSpecies(candidate.speciesId)
            return <Button key={candidate.speciesId} className={`selection result__candidate ${previewSpecies?.id === candidate.speciesId ? 'selection--active' : ''}`} disabled={(busy || associationConflict || !item) || undefined} onClick={() => selectSpecies(candidate.speciesId)}>
              <Text className='block'>{item?.commonName} · {item?.scientificName}</Text>
            </Button>
          })}
        </View>}
      </View>}
      {record.status !== 'running' && <View className='section'>
        {!showSpeciesPanel && searchAction}
        {showCatalog && <SpeciesSearch disabled={busy || associationConflict} showHint={false} onSelect={(species) => selectSpecies(species.id, 'catalog')} />}
        {(knowledgeSpecies?.knowledge || speciesSampleImageUrl(knowledgeSpecies)) && <>
          <Button className='text-button' onClick={() => setShowKnowledge(!showKnowledge)}>{showKnowledge ? '收起物种简介' : '查看物种简介'}</Button>
          {showKnowledge && <View className='result__knowledge'><SpeciesKnowledge species={knowledgeSpecies} showSafetyNotice={false} /></View>}
        </>}
      </View>}
      {linkedPlant && !associationConflict && <View className='stack section'>
        <ActionButton onClick={() => goTo(`/pages/plant/index?id=${encodeURIComponent(linkedPlant.id)}`)}>查看株记</ActionButton>
        <Text className='small muted'>已关联「{linkedPlant.nickname}」。更改或清空物种会同步此株记。</Text>
      </View>}
      {canAssociate && <View className='stack section'>
        <ActionButton onClick={() => { setShowForm(!showForm); setShowPlants(false) }}>记住这一株</ActionButton>
        <ActionButton variant='secondary' onClick={() => { setShowPlants(!showPlants); setShowForm(false) }}>关联已有植物</ActionButton>
      </View>}
      {canAssociate && showForm && <View className='form'>
        <Text className='section-title'>给这一株，一个名字</Text>
        <Text className='block small muted'>物种：{confirmed?.commonName ?? '待确认'} · 本次照片作为首次观察</Text>
        <View className='field'><Text className='field__label'>昵称 · 必填</Text><Input className='field__input' placeholder='例如：窗边的小绿' maxlength={40} value={nickname} onInput={(event) => setNickname(event.detail.value)} /></View>
        <View className='field'><Text className='field__label'>在哪里遇见 · 可选</Text><Input className='field__input' placeholder='写下地点即可，不需要定位' maxlength={80} value={location} onInput={(event) => setLocation(event.detail.value)} /></View>
        <View className='field'><Text className='field__label'>和它的故事 · 可选</Text><Textarea className='field__textarea' placeholder='也可以留到下次再写' maxlength={500} value={story} onInput={(event) => setStory(event.detail.value)} /></View>
        <View className='field'><Text className='field__label'>这一株的描述 · 可选，最多 1000 字</Text><Textarea className='field__textarea' placeholder='写下叶片、枝条等特征，仅自己可见。' maxlength={1000} value={description} onInput={(event) => setDescription(event.detail.value)} /></View>
        <ActionButton disabled={!nickname.trim()} loading={busy} onClick={() => { void createPlant() }}>创建植物档案</ActionButton>
        <Button className='text-button' onClick={() => setShowForm(false)}>暂时不建档</Button>
      </View>}
      {canAssociate && showPlants && <View className='form'>
        <Text className='section-title'>这是你记录过的哪一株？</Text>
        {snapshot.plants.length ? snapshot.plants.map((plant) => <Button key={plant.id} className='text-button selection' onClick={() => { void linkPlant(plant.id, plant.nickname) }}>{plant.nickname} →</Button>) : <StatePanel title='还没有植物档案' description='先记住这一株，以后就能继续观察它。' action='创建档案' onAction={() => { setShowPlants(false); setShowForm(true) }} />}
      </View>}
      <View className='toolbar'>
        <Text className='block small muted result__safety'>{brand.recognitionSafetyNotice}</Text>
        <Button className='text-button' onClick={() => {
          void showDialog(dialogCopy.recognition)
        }}
        >{dialogCopy.recognition.title}</Button>
        <Button className='text-button' disabled={busy || undefined} onClick={() => { void perform(async () => {
          const answer = await showDialog(dialogCopy.removeRecognition)
          if (answer.confirm) { await journal.remove('recognitions', record.id); goTab('footprints') }
        }) }}
        >删除这条足迹</Button>
      </View>
    </View></>
  )
}
