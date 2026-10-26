import assert from 'node:assert/strict'
import { test } from 'node:test'
import { downloadOptions, parseResponse, transferFailure, uploadOptions } from '../src/services/file-transfer'

test('file transfers disable cookies but retain explicit bearer authentication', () => {
  const upload = uploadOptions('https://api.example/api', { id: 'request-one', uri: 'blob:local', origin: 'local' }, 'synthetic-token')
  const download = downloadOptions('https://api.example/api', 'private-photo', 'synthetic-token')
  assert.equal(upload.withCredentials, false)
  assert.equal(download.withCredentials, false)
  assert.equal(upload.header?.Authorization, 'Bearer synthetic-token')
  assert.equal(download.header?.Authorization, 'Bearer synthetic-token')
  assert.equal(upload.name, 'file')
  assert.deepEqual(upload.formData, { requestId: 'request-one' })
  assert.equal(upload.header?.['Content-Type'], undefined) // Browser chooses multipart boundary.
  assert.equal(download.header?.['X-Photo-Id'], 'private-photo')
  assert.equal(download.url.includes('private-photo'), false)
  assert.equal(download.url.includes('synthetic-token'), false)
})

test('upload JSON response and ordinary JSON API response share decoding', () => {
  const value = { id: 'photo-one', uri: '', origin: 'remote' }
  assert.deepEqual(parseResponse(200, JSON.stringify(value)), value)
  assert.deepEqual(parseResponse(202, value), value)
})

test('HTML proxy errors retain useful size, auth, and service diagnostics', () => {
  assert.throws(() => parseResponse(413, '<html>too large</html>'), /900 KiB/)
  assert.throws(() => parseResponse(401, '<html>unauthorized</html>'), /登录/)
  assert.throws(() => parseResponse(502, '<html>bad gateway</html>'), /服务暂不可用/)
  assert.throws(() => parseResponse(200, '<html>not JSON</html>'), /响应异常/)
  assert.throws(() => parseResponse(429, { message: '稍后重试' }), /稍后重试/)
})

test('transfer errors are platform-specific and never echo raw network details', () => {
  assert.match(transferFailure('upload', { errMsg: 'uploadFile:fail timeout' }, true).message, /超时/)
  assert.match(transferFailure('upload', { errMsg: 'uploadFile:fail abort' }, true).message, /取消/)
  assert.match(transferFailure('upload', { errMsg: 'file not found' }, true).message, /重新选择/)
  const h5 = transferFailure('upload', { errMsg: 'private-token-and-url' }, true).message
  assert.match(h5, /刷新页面/)
  assert.equal(h5.includes('private-token'), false)
  assert.equal(h5.includes('合法域名'), false)
  assert.match(transferFailure('upload', {}, false).message, /上传合法域名/)
  assert.match(transferFailure('download', {}, false).message, /下载合法域名/)
})
