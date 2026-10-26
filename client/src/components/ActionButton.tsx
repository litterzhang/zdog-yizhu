import { Button, Image } from '@tarojs/components'
import type { PropsWithChildren } from 'react'

interface Props extends PropsWithChildren {
  onClick: () => void
  variant?: 'primary' | 'secondary' | 'quiet'
  disabled?: boolean
  loading?: boolean
  icon?: string
}

export function ActionButton({ children, onClick, variant = 'primary', disabled, loading, icon }: Props) {
  const isDisabled = Boolean(disabled || loading)
  return (
    <Button
      className={`button button--${variant}${isDisabled ? ' button--disabled' : ''}`}
      hoverClass='button--pressed'
      onClick={onClick}
      disabled={isDisabled || undefined}
      loading={loading || undefined}
    >
      {icon && <Image className='icon' src={`/assets/line/${icon}.png`} ariaLabel='' />}
      {children}
    </Button>
  )
}
