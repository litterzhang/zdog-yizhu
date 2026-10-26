import { defineConfig, type UserConfigExport } from '@tarojs/cli'
import { existsSync, realpathSync } from 'node:fs'
import { isAbsolute, relative, resolve } from 'node:path'
import devConfig from './dev'
import prodConfig from './prod'
import { brand } from '../src/config/brand'
import { version } from '../package.json'

export default defineConfig<'webpack5'>((merge) => {
  const outputRoot = process.env.TARO_ENV === 'h5' ? 'dist/h5' : 'dist/weapp'
  const useCI = process.argv.some((arg) => arg === '--upload' || arg === '--preview')
  const clientRoot = resolve(__dirname, '..')
  // Taro only exposes TARO_APP_* env entries. Read CI-only values in Node,
  // without giving private-key paths a frontend-exposed prefix.
  if (useCI && existsSync(resolve(clientRoot, '.env.local'))) {
    require('node:process').loadEnvFile(resolve(clientRoot, '.env.local'))
  }
  let privateKeyPath = process.env.MINIPROGRAM_PRIVATE_KEY_PATH || ''
  if (useCI) {
    if (!/^wx[0-9a-f]{16}$/.test(process.env.TARO_APP_ID || '') || !privateKeyPath) {
      throw new Error('Set a real TARO_APP_ID and MINIPROGRAM_PRIVATE_KEY_PATH in client/.env.local.')
    }
    const key = realpathSync(resolve(clientRoot, privateKeyPath))
    for (const directory of ['src', 'dist']) {
      const fromBundle = relative(resolve(clientRoot, directory), key)
      if (fromBundle !== '..' && !fromBundle.startsWith('../') && !fromBundle.startsWith('..\\') && !isAbsolute(fromBundle)) {
        throw new Error('Keep the code-upload key out of client/src and client/dist. Use client/.secrets/ or an external directory.')
      }
    }
    privateKeyPath = key
  }
  const base: UserConfigExport<'webpack5'> = {
    projectName: 'yizhu',
    date: '2026-09-20',
    designWidth: 375,
    deviceRatio: { 375: 2, 640: 1.17, 750: 1, 828: 0.905 },
    sourceRoot: 'src',
    outputRoot,
    framework: 'react',
    compiler: 'webpack5',
    plugins: useCI ? [[
      '@tarojs/plugin-mini-ci',
      {
        weapp: {
          appid: process.env.TARO_APP_ID,
          privateKeyPath,
          ignores: ['**/*.map', '**/.env*', '**/*.pem', '**/*.key', '**/.secrets/**'],
        },
        version: process.env.MINIPROGRAM_VERSION || version,
        desc: process.env.MINIPROGRAM_DESCRIPTION || 'Personal mock sample',
      },
    ]] : [],
    cache: { enable: false },
    copy: { patterns: [{ from: 'src/assets', to: `${outputRoot}/assets` }], options: {} },
    h5: {
      htmlPluginOption: { title: brand.name },
      publicPath: '/',
      staticDirectory: 'static',
      router: { mode: 'browser' },
      output: {
        filename: 'js/[name].[contenthash:8].js',
        chunkFilename: 'js/[name].[contenthash:8].js',
      },
      miniCssExtractPluginOption: {
        ignoreOrder: true,
        filename: 'css/[name].[contenthash:8].css',
        chunkFilename: 'css/[name].[contenthash:8].css',
      },
      devServer: { host: '127.0.0.1', port: 10086 },
    },
    mini: {
      postcss: {
        pxtransform: { enable: true, config: {} },
        cssModules: { enable: false },
      },
    },
  }
  return merge({}, base, process.env.NODE_ENV === 'development' ? devConfig : prodConfig)
})
