import { useState } from 'react'
import { Button, Text, View } from '@tarojs/components'
import { PhotoThumb } from '../../components/PhotoThumb'
import { StatePanel } from '../../components/StatePanel'
import { useJournal } from '../../hooks/useJournal'
import { useAccount } from '../../hooks/useAccount'
import { hasVerifiedAccount } from '../../services/account'
import { JournalStatus } from '../../components/JournalStatus'
import { goHome, goTo } from '../../services/navigation'
import { groupRecognitions, recognitionState, recognitionTitle, timeLabel } from '../../services/presentation'
import './index.scss'

export default function FootprintsPage() {
  const snapshot = useJournal()
  const session = useAccount()
  const [filter, setFilter] = useState<'all' | 'unconfirmed'>('all')

  if (!hasVerifiedAccount(session)) {
    return <View className='page tab-page footprints'><JournalStatus /></View>
  }

  const records = snapshot.recognitions.filter((record) => filter === 'all' || !record.confirmedSpeciesId)
  const groups = groupRecognitions(records)

  return (
    <View className='page tab-page footprints'>
      <JournalStatus />
      <View className='section-meta'>
        <Text>我的识别记录</Text>
        <Text className='section-meta__count'>{snapshot.recognitions.length} 次相遇</Text>
      </View>
      <View className='filter-row'>
        <Button className={`filter-button ${filter === 'all' ? 'filter-button--active' : ''}`} onClick={() => setFilter('all')}>全部</Button>
        <Button className={`filter-button ${filter === 'unconfirmed' ? 'filter-button--active' : ''}`} onClick={() => setFilter('unconfirmed')}>待确认</Button>
      </View>
      {!records.length && <StatePanel
        title={snapshot.recognitions.length ? '这一页，暂时没有记录' : '第一步，从发现开始'}
        description={snapshot.recognitions.length ? '换个筛选，看看其他相遇。' : '在「发现」选择照片并识别后，就能在这里再次找到它。'}
        action={snapshot.recognitions.length ? '查看全部足迹' : '去发现一株植物'}
        onAction={snapshot.recognitions.length ? () => setFilter('all') : goHome}
      />}
      {groups.map((group) => <View className='footprints__group' key={group.date}>
        <View className='footprints__date'><View className='footprints__date-dot' /><Text>{group.date}</Text></View>
        {group.records.map((record) => {
          const saved = Boolean(record.linkedPlantId) || record.associationConflict
          return <Button key={record.id} className='footprints__entry' hoverClass='button--pressed' onClick={() => goTo(`/pages/result/index?id=${encodeURIComponent(record.id)}`)}>
            <PhotoThumb photo={record.photo} className='footprints__photo' />
            <View className='grow'>
              <View className='row between footprints__entry-heading'>
                <Text className='footprints__name'>{recognitionTitle(record)}</Text>
                <Text className='footprints__arrow'>›</Text>
              </View>
              <Text className='block footprints__time'>{timeLabel(record.createdAt)} · {record.photo.origin === 'demo' ? '示例照片' : '本次选图'}</Text>
              <View className='footprints__labels'>
                <Text className={`footprints__status footprints__status--${record.status}`}>{recognitionState(record)}</Text>
                {saved && <Text className='footprints__saved'>已记入株记</Text>}
              </View>
            </View>
          </Button>
        })}
      </View>)}
      <Text className='block subtle-footer'>私有保存 · 候选需要确认 · 不自动关联植物个体</Text>
    </View>
  )
}
