export interface DialogOptions {
  title: string
  content: string
  showCancel?: boolean
  confirmText?: string
  cancelText?: string
  tone?: 'normal' | 'danger'
}

export interface DialogResult {
  confirm: boolean
  cancel: boolean
}

export interface DialogState extends Required<DialogOptions> {
  id: number
}

export type ShowDialog = (options: DialogOptions) => Promise<DialogResult>

let nextDialogId = 0

// Each page owns its controller. A hidden page can never retain a pending
// destructive confirmation or open another dialog from a late async callback.
export class DialogController {
  private state: DialogState | null = null
  private active = true
  private resolve: ((result: DialogResult) => void) | null = null
  private listeners = new Set<() => void>()

  getSnapshot = () => this.state
  subscribe = (listener: () => void) => {
    this.listeners.add(listener)
    return () => { this.listeners.delete(listener) }
  }

  open: ShowDialog = (options) => {
    // Do not share a confirmation promise: two callers could perform the same
    // destructive action twice. Ignore overlapping requests as cancellations.
    if (!this.active || this.state) return Promise.resolve({ confirm: false, cancel: true })
    const showCancel = options.showCancel ?? true
    this.state = {
      ...options,
      id: ++nextDialogId,
      showCancel,
      confirmText: options.confirmText ?? (showCancel ? '确认' : '知道了'),
      cancelText: options.cancelText ?? '取消',
      tone: options.tone ?? 'normal',
    }
    const result = new Promise<DialogResult>((resolve) => { this.resolve = resolve })
    this.publish()
    return result
  }

  settle = (id: number, confirm: boolean) => {
    if (this.state?.id !== id) return
    const resolve = this.resolve
    this.state = null
    this.resolve = null
    this.publish()
    resolve?.({ confirm, cancel: !confirm })
  }

  activate = () => { this.active = true }

  deactivate = () => {
    this.active = false
    if (this.state) this.settle(this.state.id, false)
  }

  private publish() {
    this.listeners.forEach((listener) => listener())
  }
}
