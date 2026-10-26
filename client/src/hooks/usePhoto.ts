import { useEffect, useState } from 'react'
import type { Photo } from '../types/domain'
import { privatePhoto, retryPrivatePhoto } from '../services/transport'

export function usePhoto(photo: Photo) {
  const { id, uri: sourceUri, origin } = photo
  const [attempt, setAttempt] = useState(0)
  const [state, setState] = useState({ id: '', uri: '', error: '' })
  useEffect(() => {
    let active = true
    setState({ id, uri: '', error: '' })
    void Promise.resolve().then(() => privatePhoto({ id, uri: sourceUri, origin })).then((uri) => {
      if (active) setState({ id, uri, error: '' })
    }).catch(() => {
      if (active) setState({ id, uri: '', error: '照片未加载 · 轻点重试' })
    })
    return () => { active = false }
  }, [id, sourceUri, origin, attempt])
  return {
    uri: photo.origin !== 'remote' ? photo.uri : state.id === photo.id ? state.uri : '',
    error: state.id === photo.id ? state.error : '',
    retry: () => { retryPrivatePhoto(photo.id); setAttempt((value) => value + 1) },
  }
}
