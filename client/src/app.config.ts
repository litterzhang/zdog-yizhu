import { brand } from './config/brand'
import { navigationTabs } from './config/navigation'

export default defineAppConfig({
  pages: [
    ...navigationTabs.map((tab) => tab.pagePath),
    'pages/result/index',
    'pages/plant/index',
    'pages/login/index',
    'pages/catalog/index',
  ],
  window: {
    navigationBarTitleText: brand.name,
    navigationBarBackgroundColor: '#FFFFFF',
    navigationBarTextStyle: 'black',
    backgroundColor: '#FFFFFF',
    backgroundTextStyle: 'dark',
  },
  tabBar: {
    color: '#667168',
    selectedColor: '#496B50',
    backgroundColor: '#FFFFFF',
    borderStyle: 'white',
    list: navigationTabs.map((tab) => ({
      pagePath: tab.pagePath,
      text: tab.text,
      iconPath: `assets/tabs/${tab.icon}.png`,
      selectedIconPath: `assets/tabs/${tab.icon}-selected.png`,
    })),
  },
})
