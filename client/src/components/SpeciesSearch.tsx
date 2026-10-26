import { useEffect, useRef, useState } from 'react'
import { Button, Input, Text, View } from '@tarojs/components'
import { searchSpecies } from '../services/plant-catalog'
import { rememberSpecies } from '../services/catalog'
import { speciesSearchName } from '../services/presentation'
import type { Species } from '../types/domain'
import { StatePanel } from './StatePanel'

export function SpeciesSearch({ onSelect, disabled = false, actionLabel = '选择此物种', showHint = true }: {
  onSelect: (species: Species) => void; disabled?: boolean; actionLabel?: string; showHint?: boolean
}) {
  const [query, setQuery] = useState('')
  const [items, setItems] = useState<Species[]>([])
  const [total, setTotal] = useState<number | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const request = useRef(0)
  useEffect(() => () => { request.current++ }, [])

  async function search(more = false) {
    if (!query.trim() || disabled || loading) return
    const ticket = ++request.current
    const offset = more ? items.length : 0
    setError(''); setLoading(true)
    if (!more) { setItems([]); setTotal(null) }
    try {
      const response = await searchSpecies({ q: query.trim(), offset })
      if (ticket !== request.current) return
      setItems((previous) => more ? [...previous, ...response.items] : response.items)
      setTotal(response.total)
    } catch (err) {
      if (ticket === request.current) setError(err instanceof Error ? err.message : '搜索失败，请重试。')
    } finally { if (ticket === request.current) setLoading(false) }
  }

  return <View className='species-search'>
    {showHint && <Text className='block small muted'>输入中文名或学名查找。仅展示有中文名的植物；未找到时可先保留待确认。</Text>}
    <View className='catalog-search'>
      <Input className='field__input grow' placeholder='例如：银杏 / Ginkgo biloba' maxlength={80} value={query} onInput={(event) => {
        request.current++; setLoading(false); setQuery(event.detail.value); setItems([]); setTotal(null); setError('')
      }} onConfirm={() => { void search() }}
      />
      <Button className='text-button' disabled={(loading || disabled || !query.trim()) || undefined} onClick={() => { void search() }}>搜索</Button>
    </View>
    {loading && <Text className='block small muted'>正在查找…</Text>}
    {error && <StatePanel title='搜索未完成' description={error} action='重试' onAction={() => { void search() }} />}
    {!loading && !error && total === 0 && <Text className='block muted'>名录中暂未找到，可以换用学名，或保留待确认。</Text>}
    {items.map((item) => {
      const name = speciesSearchName(item)
      return <Button key={item.id} className='text-button catalog-entry' disabled={disabled || undefined} onClick={() => {
        rememberSpecies(item); onSelect(item)
      }}
      >
        <Text className='block'>{name}</Text>
        <Text className='block small muted'>{item.scientificName} · {item.taxonomyChinese?.family ?? item.taxonomy?.family ?? '科未知'}</Text>
        {name === item.scientificName && <Text className='block small muted'>中文名待补充</Text>}
        <Text className='block small'>{actionLabel}</Text>
      </Button>
    })}
    {total !== null && items.length < total && items.length > 0 && <Button className='text-button' disabled={(loading || disabled) || undefined} onClick={() => { void search(true) }}>更多物种</Button>}
  </View>
}
