import { createApp } from 'vue'
import App from './App.vue'
import router from './router'
import 'katex/dist/katex.min.css'
import './styles/main.css'
import { zhPunct } from './utils/zhPunct.js'

const app = createApp(App)
// 模板插值在编译期被包成 $zh(...) (plugins/zh-punct-template.js), 页面上的正文标点显示成中文
app.config.globalProperties.$zh = zhPunct
app.use(router).mount('#app')
