import assert from 'node:assert/strict'
import { test } from 'node:test'
import { dialogCopy } from '../src/config/dialogs'

test('all dialog copy omits implementation details and demo wording', () => {
  for (const [key, entry] of Object.entries(dialogCopy)) {
    const dialog = typeof entry === 'function' ? entry('窗边的小绿') : entry
    assert.ok(dialog.title.trim(), key)
    assert.ok(dialog.content.trim(), key)
    assert.doesNotMatch(`${dialog.title}\n${dialog.content}`, /BioCLIP|Pl@ntNet|demo|演示|示例|算法|模型|版本|服务：/i, key)
  }
})

test('shortened dialogs retain destructive warnings, privacy and explicit association', () => {
  for (const dialog of [
    dialogCopy.clearData, dialogCopy.removeObservation,
    dialogCopy.removeRecognition, dialogCopy.removePlant('窗边的小绿'),
  ]) {
    assert.equal(dialog.tone, 'danger')
    assert.notEqual(dialog.showCancel, false)
    assert.match(dialog.content, /无法撤销/)
    assert.ok(dialog.confirmText)
  }
  assert.match(dialogCopy.clearData.content, /不会删除手机相册原图或注销账号/)
  assert.match(dialogCopy.signOut.content, /记录与照片保留/)
  assert.match(dialogCopy.saveObservation.content, /上传/)
  assert.match(dialogCopy.recognition.content, /不能据此食用、用药/)
  assert.match(dialogCopy.linkPlant('窗边的小绿').content, /「窗边的小绿」/)
  assert.match(dialogCopy.linkPlant('窗边的小绿').content, /同一株植物，而不只是同一种/)
  assert.match(dialogCopy.photoCredits.content, /Tangopaso.*Wikimedia Commons/)
})
