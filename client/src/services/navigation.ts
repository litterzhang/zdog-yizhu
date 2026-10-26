import Taro from '@tarojs/taro'
import { navigationTabs, type TabKey } from '../config/navigation'

export function goTo(url: string) {
  const tab = navigationTabs.find((item) => `/${item.pagePath}` === url.split('?')[0])
  if (tab) {
    goTab(tab.key)
    return
  }
  void Taro.navigateTo({ url }).catch(() => {
    void Taro.showToast({ title: '页面未打开，请重试', icon: 'none' })
  })
}

export function goHome() {
  goTab('discover')
}

export function goTab(key: TabKey) {
  const tab = navigationTabs.find((item) => item.key === key)!
  void Taro.switchTab({ url: `/${tab.pagePath}` }).catch(() => {
    void Taro.showToast({ title: '切换失败，请重试', icon: 'none' })
  })
}
