import { useEffect, useState, useSyncExternalStore } from 'react'
import Taro, { useDidHide, useDidShow } from '@tarojs/taro'
import { Dialog } from '../components/Dialog'
import { DialogController } from '../services/dialog'

export function useDialog(tabPage = false) {
  const [controller] = useState(() => new DialogController())
  const state = useSyncExternalStore(controller.subscribe, controller.getSnapshot)
  useDidShow(controller.activate)
  useDidHide(controller.deactivate)
  useEffect(() => {
    controller.activate()
    return controller.deactivate
  }, [controller])

  const open = Boolean(state)
  useEffect(() => {
    if (!open || !tabPage) return
    // Native tab bars are above page content. Hide them only while a tab-page
    // dialog is open, restoring them on dismissal, navigation or unmount.
    void Taro.hideTabBar({ animation: false }).catch(() => {})
    return () => { void Taro.showTabBar({ animation: false }).catch(() => {}) }
  }, [open, tabPage])

  return {
    showDialog: controller.open,
    dialog: state ? <Dialog key={state.id} state={state} onClose={controller.settle} /> : null,
  }
}
