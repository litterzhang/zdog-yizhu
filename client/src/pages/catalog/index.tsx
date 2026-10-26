import { useEffect, useRef, useState } from 'react'
import { Button, Input, Text, View } from '@tarojs/components'
import Taro, { useRouter } from '@tarojs/taro'
import { StatePanel } from '../../components/StatePanel'
import { SpeciesKnowledge } from '../../components/SpeciesKnowledge'
import {
  browseTaxa, catalogInfo, rankNames, readSpecies, searchSpecies,
  type CatalogInfo, type TaxonNode,
} from '../../services/plant-catalog'
import type { Species } from '../../types/domain'
import { catalogRouteId } from '../../services/catalog'
import { catalogDetailSources } from '../../services/presentation'
import './index.scss'

export default function CatalogPage() {
  const { params } = useRouter()
  const [info, setInfo] = useState<CatalogInfo | null>(null)
  const [trail, setTrail] = useState<TaxonNode[]>([])
  const [draft, setDraft] = useState('')
  const [query, setQuery] = useState('')
  const [items, setItems] = useState<Array<Species | TaxonNode>>([])
  const [total, setTotal] = useState(0)
  const [selected, setSelected] = useState<Species | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [retry, setRetry] = useState(0)
  const ticket = useRef(0)
  const parent = trail[trail.length - 1]
  const speciesMode = Boolean(query) || parent?.rank === 'genus'

  useEffect(() => {
    let active = true
    void catalogInfo().then((value) => { if (active) setInfo(value) }).catch(() => {})
    return () => { active = false }
  }, [retry])

  useEffect(() => {
    const current = ++ticket.current
    setItems([]); setTotal(0); setError(''); setLoading(true); setSelected(null)
    const action = params.id
      ? readSpecies(catalogRouteId(params.id)).then((value) => { if (current === ticket.current) setSelected(value) })
      : (speciesMode ? searchSpecies({ q: query, parent: parent?.id })
        : browseTaxa({ parent: parent?.id })).then((value) => {
        if (current === ticket.current) { setItems(value.items); setTotal(value.total) }
      })
    void action.catch((err: Error) => { if (current === ticket.current) setError(err.message) })
      .finally(() => { if (current === ticket.current) setLoading(false) })
    // This ref is a request-generation counter, not a DOM element.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    return () => { ticket.current++ }
  }, [params.id, query, parent?.id, speciesMode, retry])

  async function more() {
    if (loading) return
    const current = ++ticket.current
    setLoading(true); setError('')
    try {
      const options = { parent: parent?.id, offset: items.length }
      const response = await (speciesMode ? searchSpecies({ ...options, q: query }) : browseTaxa(options))
      if (current === ticket.current) {
        setItems((previous) => [...previous, ...response.items]); setTotal(response.total)
      }
    } catch (err) { if (current === ticket.current) setError(err instanceof Error ? err.message : '加载失败，请重试。') }
    finally { if (current === ticket.current) setLoading(false) }
  }

  return <View className='page catalog-page'>
    {info?.available && <Text className='block small muted'>当前名录 {info.plantCount.toLocaleString()} 个植物参考条目 · 不是世界植物全名录，也不保证每种都能准确识别。</Text>}
    {info?.chineseNameCount !== undefined && <Text className='block small muted'>当前展示 {(info.visiblePlantCount ?? info.chineseNameCount).toLocaleString()} 个有中文名的条目；缺少中文名的条目暂不展示。</Text>}
    {!params.id && <>
      <View className='catalog-search'>
        <Input className='field__input grow' placeholder='中文名或学名' maxlength={80} value={draft} onInput={(event) => setDraft(event.detail.value)} onConfirm={() => { setQuery(draft.trim()); setSelected(null) }} />
        <Button className='text-button' onClick={() => { setQuery(draft.trim()); setSelected(null) }}>搜索</Button>
      </View>
      <View className='catalog-trail'>
        <Button className='text-button small' onClick={() => { setTrail([]); setQuery(''); setDraft(''); setSelected(null) }}>全部植物</Button>
        {trail.map((node, index) => <Button key={node.id} className='text-button small' onClick={() => { setTrail(trail.slice(0, index + 1)); setSelected(null) }}>› {node.chineseName ?? node.name}（{rankNames[node.rank]}）</Button>)}
      </View>
    </>}
    {error && <StatePanel title='名录未加载完成' description={error} action='重试' onAction={() => { setRetry((value) => value + 1) }} />}
    {selected ? <View className='catalog-detail'>
      {!params.id && <Button className='text-button' onClick={() => setSelected(null)}>收起物种详情</Button>}
      <Text className='title'>{selected.commonName}</Text>
      <Text className='block muted'>学名：{selected.scientificName}</Text>
      {selected.commonName === selected.scientificName && <Text className='block small muted'>中文名待补充，暂以学名区分。</Text>}
      {!!selected.chineseAliases?.length && <Text className='block small muted'>中文别名：{selected.chineseAliases.join('、')}</Text>}
      <Text className='block small muted'>名称不是个体身份；名录信息不等于照片鉴定结论。</Text>
      <View className='section'>
        {(['order', 'family', 'genus'] as const).map((rank) => <View key={rank} className='catalog-taxonomy'><Text className='catalog-taxonomy__label'>{rankNames[rank]}</Text><View className='catalog-taxonomy__value'><Text className='block'>{selected.taxonomyChinese?.[rank] || selected.taxonomy?.[rank] || '未知'}</Text>{selected.taxonomyChinese?.[rank] && <Text className='block small muted'>{selected.taxonomy?.[rank]}</Text>}</View></View>)}
        <View className='catalog-taxonomy'><Text className='catalog-taxonomy__label'>种</Text><Text className='catalog-taxonomy__value'>{selected.commonName}</Text></View>
      </View>
      {!!selected.taxonomyAlternatives?.length && <Text className='block small muted'>原始名录存在其他分类路径；此处展示第一条来源记录，不自动判断哪条为最新接受分类。</Text>}
      <SpeciesKnowledge species={selected} showSafetyNotice={false} />
      <View className='notice'>分类来自版本化公开参考名录，可能存在旧分类或同物异名。不据此食用、用药或判断树龄。</View>
      {catalogDetailSources(selected).map((source) => <Button key={source.url} className='text-button small' onClick={() => { void Taro.setClipboardData({ data: source.url }) }}>来源：{source.title} · 复制出处</Button>)}
      {selected.nameSources?.map((source) => <View key={`${source.url}:${source.sourceName}`}>
        {source.sourceName && source.sourceName !== selected.commonName && <Text className='block small muted'>来源中的中文名：{source.sourceName}</Text>}
        <Button className='text-button small' onClick={() => { void Taro.setClipboardData({ data: source.url }) }}>中文名来源：{source.title} · {source.version} · 复制出处</Button>
      </View>)}
    </View> : <>
      {!loading && !error && !items.length && <StatePanel title='这里暂时没有条目' description='目前仅展示有中文名的植物。可以换个名称搜索，或先将植物记录为待确认。' action='浏览全部植物' onAction={() => {
        setTrail([]); setQuery(''); setDraft('')
      }}
      />}
      {items.map((item) => 'rank' in item
        ? <Button key={item.id} className='text-button catalog-entry' onClick={() => { setTrail([...trail, item]); setSelected(null) }}><Text className='block'>{item.chineseName ?? item.name}{item.chineseName ? '' : ` · ${rankNames[item.rank]}`}</Text>{item.chineseName && <Text className='block small muted'>{item.name}</Text>}<Text className='small muted'>{item.count} 个参考条目 ›</Text></Button>
        : <Button key={item.id} className='text-button catalog-entry' onClick={() => setSelected(item)}><Text className='block'>{item.commonName}</Text><Text className='block small muted'>{item.scientificName} · {item.taxonomyChinese?.family ?? item.taxonomy?.family ?? '科未知'}</Text>{item.commonName === item.scientificName && <Text className='block small muted'>中文名待补充</Text>}</Button>)}
      {items.length > 0 && <Text className='block small muted'>已加载 {items.length} / {total} 条</Text>}
      {items.length < total && <Button className='text-button' disabled={loading || undefined} onClick={() => { void more() }}>加载更多</Button>}
    </>}
    {loading && <Text className='block muted'>正在加载名录…</Text>}
    {!selected && <Text className='block subtle-footer'>{info?.source ?? '植物参考名录'} · 物种知识与私有株记分开保存</Text>}
  </View>
}
