import { useEffect, useRef } from 'react'
import { Button, PageMeta, RootPortal, ScrollView, Text, View } from '@tarojs/components'
import Taro from '@tarojs/taro'
import type { DialogState } from '../services/dialog'
import './Dialog.scss'

if (process.env.TARO_ENV === 'h5') require('./Dialog.h5.scss')

interface Props {
  state: DialogState
  onClose: (id: number, confirm: boolean) => void
}

export function Dialog({ state, onClose }: Props) {
  const panelRef = useRef<HTMLElement | null>(null)
  useEffect(() => {
    if (process.env.TARO_ENV !== 'h5') return
    // H5-only accessibility through component refs, never browser globals.
    const panel = panelRef.current
    if (!panel) return
    const buttons = Array.from(panel.querySelectorAll<HTMLElement>('.yizhu-dialog__button'))
    panel.setAttribute('tabindex', '-1')
    buttons.forEach((button) => {
      button.setAttribute('tabindex', '0')
      button.setAttribute('role', 'button')
    })
    let active = true
    // Taro H5 upgrades custom elements asynchronously. Focusing before their
    // first render can lose focus when Stencil installs the slotted content.
    void Promise.all([panel.parentElement, panel, ...panel.querySelectorAll('*')].map((element) =>
      (element as (HTMLElement & { componentOnReady?: () => Promise<unknown> }) | null)?.componentOnReady?.(),
    )).then(() => Taro.nextTick(() => { if (active) panel.focus({ preventScroll: true }) }))
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.preventDefault()
        onClose(state.id, false)
      } else if (event.key === 'Tab') {
        event.preventDefault()
        const index = buttons.indexOf(event.target as HTMLElement)
        const next = index < 0 ? (event.shiftKey ? buttons.length - 1 : 0)
          : (index + (event.shiftKey ? -1 : 1) + buttons.length) % buttons.length
        buttons[next]?.focus({ preventScroll: true })
      } else if ((event.key === 'Enter' || event.key === ' ') && buttons.includes(event.target as HTMLElement)) {
        event.preventDefault()
        ;(event.target as HTMLElement).click()
      }
    }
    panel.addEventListener('keydown', onKeyDown)
    return () => {
      active = false
      panel.removeEventListener('keydown', onKeyDown)
    }
  }, [state.id, onClose])

  const overlay = <View className={`yizhu-dialog yizhu-dialog--${state.tone}`}>
    <View className='yizhu-dialog__backdrop' catchMove onTouchMove={(event) => event.preventDefault()} />
    <View
      className='yizhu-dialog__panel'
      ref={panelRef}
      role='dialog'
      ariaLabel={state.title}
      aria-modal='true'
      aria-labelledby={`dialog-title-${state.id}`}
      aria-describedby={`dialog-content-${state.id}`}
    >
      <View className='yizhu-dialog__heading'>
        <View className='yizhu-dialog__sprig' aria-hidden='true'><View /><View /></View>
        <Text className='yizhu-dialog__title' id={`dialog-title-${state.id}`}>{state.title}</Text>
      </View>
      <ScrollView className='yizhu-dialog__body' scrollY enhanced showScrollbar={false}>
        <Text className='yizhu-dialog__content' id={`dialog-content-${state.id}`}>{state.content}</Text>
      </ScrollView>
      <View className='yizhu-dialog__actions'>
        {state.showCancel && <Button
          className='yizhu-dialog__button yizhu-dialog__button--cancel'
          hoverClass='yizhu-dialog__button--pressed'
          onClick={() => onClose(state.id, false)}
        >{state.cancelText}</Button>}
        <Button
          className='yizhu-dialog__button yizhu-dialog__button--confirm'
          hoverClass='yizhu-dialog__button--pressed'
          onClick={() => onClose(state.id, true)}
        >{state.confirmText}</Button>
      </View>
    </View>
  </View>

  return process.env.TARO_ENV === 'weapp'
    ? <><PageMeta pageStyle='overflow: hidden;' /><RootPortal>{overlay}</RootPortal></>
    : overlay
}
