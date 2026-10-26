import { Button } from '@tarojs/components'
import { useDialog } from '../hooks/useDialog'
import type { ShowDialog } from '../services/dialog'
import { dialogCopy } from '../config/dialogs'

export function showPhotoCredits(showDialog: ShowDialog) {
  return showDialog(dialogCopy.photoCredits)
}

export function PhotoCredits() {
  const { showDialog, dialog } = useDialog()
  return <>{dialog}<Button className='text-button small' onClick={() => { void showPhotoCredits(showDialog) }}>照片来源与说明</Button></>
}
