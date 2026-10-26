import type { PropsWithChildren } from 'react'
import { useDidHide, useDidShow } from '@tarojs/taro'
import { accountSession } from './services/session'
import { journal } from './services/journal'
import './app.scss'

export default function App({ children }: PropsWithChildren) {
  useDidShow(() => { void accountSession.restore() })
  useDidHide(() => { journal.pause() })
  return children
}
