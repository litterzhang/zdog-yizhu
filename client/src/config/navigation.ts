export const navigationTabs = [
  { key: 'discover', pagePath: 'pages/index/index', text: '发现', icon: 'discover' },
  { key: 'footprints', pagePath: 'pages/footprints/index', text: '足迹', icon: 'footprints' },
  { key: 'journal', pagePath: 'pages/journal/index', text: '株记', icon: 'journal' },
  { key: 'me', pagePath: 'pages/me/index', text: '我的', icon: 'me' },
] as const

export type TabKey = typeof navigationTabs[number]['key']
