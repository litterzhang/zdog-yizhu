// babel-preset-taro 更多选项和默认值：
// https://docs.taro.zone/docs/next/babel-config
module.exports = {
  presets: [
    ['taro', {
      framework: 'react',
      ts: true,
      compiler: 'webpack5',
      // Keep WeChat output on Taro's conservative targets, independently of
      // the modern browserslist used by H5 (which can retain ?. and ??).
      ...(process.env.TARO_ENV === 'weapp'
        ? { targets: { ios: '9', android: '5' }, ignoreBrowserslistConfig: true }
        : {}),
    }]
  ]
}
