import { useState } from 'react'
import { Button, Image, Input, Text, View } from '@tarojs/components'
import { PhotoThumb } from '../../components/PhotoThumb'
import { StatePanel } from '../../components/StatePanel'
import { useJournal } from '../../hooks/useJournal'
import { useAccount } from '../../hooks/useAccount'
import { hasVerifiedAccount } from '../../services/account'
import { JournalStatus } from '../../components/JournalStatus'
import { goHome, goTo } from '../../services/navigation'
import { dateLabel, lastObservationAt, searchPlants, speciesName } from '../../services/presentation'
import './index.scss'

export default function JournalPage() {
  const snapshot = useJournal()
  const session = useAccount()
  const [query, setQuery] = useState('')
  const [unknownOnly, setUnknownOnly] = useState(false)

  if (!hasVerifiedAccount(session)) {
    return <View className='page tab-page journal'><JournalStatus /></View>
  }

  const plants = searchPlants(snapshot.plants, query)
    .filter((plant) => !unknownOnly || !plant.speciesId)
    .sort((a, b) => Date.parse(lastObservationAt(b, snapshot)) - Date.parse(lastObservationAt(a, snapshot)))

  return (
    <View className='page tab-page journal'>
      <JournalStatus />
      <View className='journal__search'>
        <View className='journal__search-icon' />
        <Input className='journal__search-input' value={query} maxlength={80} placeholder='搜索昵称或物种' onInput={(event) => setQuery(event.detail.value)} ariaLabel='搜索已记住的植物' />
        {query && <Button className='journal__search-clear' onClick={() => setQuery('')}>清除</Button>}
      </View>
      <View className='filter-row'>
        <Button className={`filter-button ${!unknownOnly ? 'filter-button--active' : ''}`} onClick={() => setUnknownOnly(false)}>全部 · {snapshot.plants.length}</Button>
        <Button className={`filter-button ${unknownOnly ? 'filter-button--active' : ''}`} onClick={() => setUnknownOnly(true)}>物种待确认</Button>
      </View>
      {!plants.length && <StatePanel
        title={snapshot.plants.length ? '还没找到这一株' : '为一株植物，留一个位置'}
        description={snapshot.plants.length ? '试试其他昵称或物种名称，也可以清除筛选。' : '在识别结果里点「记住这一株」，它就会来到这里。认不出物种也可以记录。'}
        action={snapshot.plants.length ? '清除搜索与筛选' : '去发现'}
        onAction={snapshot.plants.length ? () => { setQuery(''); setUnknownOnly(false) } : goHome}
      />}
      <View className='journal__grid'>
        {plants.map((plant) => <Button
          key={plant.id}
          className='journal__plant'
          hoverClass='button--pressed'
          onClick={() => goTo(`/pages/plant/index?id=${encodeURIComponent(plant.id)}`)}
        >
          <View className='journal__cover'>
            <PhotoThumb photo={plant.cover} className='journal__photo' />
            <View className='journal__saved-mark'><Image className='journal__saved-icon' src='/assets/tabs/journal-selected.png' ariaLabel='已记住的植物' /></View>
          </View>
          <Text className='journal__name'>{plant.nickname}</Text>
          <Text className='block journal__species'>{speciesName(plant.speciesId)}</Text>
          <Text className='block journal__date'>{dateLabel(lastObservationAt(plant, snapshot))}</Text>
          <Text className='block journal__caption'>最近一次观察</Text>
        </Button>)}
      </View>
      {!!plants.length && <View className='journal__closing'>
        <Image src='/assets/line/sprig.png' mode='aspectFit' className='journal__sprig' ariaLabel='' />
        <Text>有空，再去看看它们。</Text>
      </View>}
      <Text className='block subtle-footer'>私有保存 · 昵称可以修改，个体身份保持不变</Text>
    </View>
  )
}
