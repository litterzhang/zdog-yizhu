import assert from 'node:assert/strict'
import { test } from 'node:test'
import { DialogController } from '../src/services/dialog'

const options = { title: '确定是同一株吗？', content: '物种相同不代表是同一个体。' }

test('dialogs wait for explicit confirmation and preserve their content and action labels', async () => {
  const controller = new DialogController()
  const result = controller.open({ ...options, confirmText: '是同一株' })
  const state = controller.getSnapshot()!
  assert.equal(state.title, options.title)
  assert.equal(state.content, options.content)
  assert.equal(state.showCancel, true)
  assert.equal(state.cancelText, '取消')
  assert.equal(state.confirmText, '是同一株')
  assert.equal(state.tone, 'normal')
  let resolved = false
  void result.then(() => { resolved = true })
  await Promise.resolve()
  assert.equal(resolved, false)
  controller.settle(state.id, true)
  assert.deepEqual(await result, { confirm: true, cancel: false })
  assert.equal(controller.getSnapshot(), null)
})

test('danger dialogs cancel without approving destructive operations', async () => {
  const controller = new DialogController()
  const result = controller.open({ ...options, tone: 'danger', confirmText: '确认删除' })
  assert.equal(controller.getSnapshot()?.tone, 'danger')
  controller.settle(controller.getSnapshot()!.id, false)
  assert.deepEqual(await result, { confirm: false, cancel: true })
})

test('informational dialogs have one acknowledgement action by default', async () => {
  const controller = new DialogController()
  const result = controller.open({ ...options, showCancel: false })
  assert.equal(controller.getSnapshot()?.showCancel, false)
  assert.equal(controller.getSnapshot()?.confirmText, '知道了')
  controller.settle(controller.getSnapshot()!.id, true)
  assert.equal((await result).confirm, true)
})

test('overlapping requests are cancelled rather than approving two callers with one click', async () => {
  const controller = new DialogController()
  const first = controller.open(options)
  const state = controller.getSnapshot()!
  const second = controller.open({ ...options, title: '另一次操作' })
  assert.deepEqual(await second, { confirm: false, cancel: true })
  assert.equal(controller.getSnapshot(), state)
  controller.settle(state.id, true)
  assert.equal((await first).confirm, true)
})

test('old or double clicks cannot confirm a newer dialog', async () => {
  const controller = new DialogController()
  const first = controller.open(options)
  const id = controller.getSnapshot()!.id
  controller.settle(id, true)
  await first
  const second = controller.open(options)
  controller.settle(id, true)
  assert.notEqual(controller.getSnapshot()?.id, id)
  controller.settle(controller.getSnapshot()!.id, false)
  assert.equal((await second).confirm, false)
})

test('page hide/unload cancels pending work and refuses late dialog requests until shown again', async () => {
  const controller = new DialogController()
  const pending = controller.open(options)
  controller.deactivate()
  assert.equal((await pending).confirm, false)
  assert.equal(controller.getSnapshot(), null)
  assert.equal((await controller.open(options)).confirm, false)
  controller.activate()
  const resumed = controller.open(options)
  assert.ok(controller.getSnapshot())
  controller.deactivate()
  assert.equal((await resumed).confirm, false)
})

test('page controllers do not leak their dialogs or confirmations to other pages', async () => {
  const first = new DialogController()
  const second = new DialogController()
  const a = first.open(options)
  const b = second.open({ ...options, title: '另一页' })
  first.deactivate()
  assert.equal((await a).confirm, false)
  assert.equal(second.getSnapshot()?.title, '另一页')
  second.settle(second.getSnapshot()!.id, true)
  assert.equal((await b).confirm, true)
})

test('subscriptions are stable and can unsubscribe before controller cleanup', async () => {
  const controller = new DialogController()
  let changes = 0
  const unsubscribe = controller.subscribe(() => { changes++ })
  const result = controller.open(options)
  const state = controller.getSnapshot()
  assert.equal(controller.getSnapshot(), state)
  assert.equal(changes, 1)
  unsubscribe()
  controller.deactivate()
  await result
  assert.equal(changes, 1)
})
