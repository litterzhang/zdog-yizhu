import { Button, Text, View } from '@tarojs/components'
import Taro from '@tarojs/taro'
import { brand } from '../config/brand'
import type { Species } from '../types/domain'
import { speciesSampleImageUrl } from '../services/presentation'
import { SpeciesSampleImage } from './SpeciesSampleImage'

export function SpeciesKnowledge({ species, showSafetyNotice = true }: {
  species?: Species | null
  showSafetyNotice?: boolean
}) {
  const knowledge = species?.knowledge
  const sampleImageUrl = speciesSampleImageUrl(species)
  return <View className='species-knowledge'>
    <Text className='section-title'>物种简介</Text>
    {species && <Text className='block small muted'>资料对应：{species.commonName} · {species.scientificName}</Text>}
    {knowledge
      ? <>
        <Text className='block small muted'>{knowledge.isAiGenerated ? 'AI 辅助摘要' : '来源资料摘要'}{knowledge.reviewStatus !== 'reviewed' ? ' · 尚未人工校对' : ''}</Text>
        <Text className='block species-knowledge__summary'>{knowledge.summary}</Text>
      </>
      : <Text className='block small muted'>该物种的有来源简介尚未收录。可以先在株记中保存自己的描述，不会自动公开或作为物种知识。</Text>}
    {species && sampleImageUrl && <SpeciesSampleImage
      key={`${species.id}:${sampleImageUrl}`}
      url={sampleImageUrl}
      name={species.commonName}
    />}
    {knowledge && <>
      {knowledge.sources.map((source) => <Button key={source.url} className='text-button small' onClick={() => { void Taro.setClipboardData({ data: source.url }) }}>来源：{source.title} · 复制出处</Button>)}
      {showSafetyNotice && <Text className='block small muted'>{brand.recognitionSafetyNotice}</Text>}
    </>}
  </View>
}
