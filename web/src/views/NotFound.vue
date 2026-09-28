<!-- 兜底页: 地址写错, 或这一章没有内容时显示。StageTopic 找不到章节时也用它。 -->
<template>
  <div>
    <h1 class="page-title">没有这一页</h1>
    <p class="page-subtitle">地址可能写错了, 或者这一章换了位置。从下面两个入口重新找。</p>
    <div class="btn-group">
      <router-link :to="{ name: 'home' }" class="btn active">回主线总览</router-link>
      <router-link :to="{ name: 'fast-track' }" class="btn">看速成路线</router-link>
    </div>
  </div>
</template>
