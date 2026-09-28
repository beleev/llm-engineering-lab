<!--
  步进控件: 配合 useStepper 使用。 <StepPlayer :stepper="s" :label="`t = ${s.step.value}`" />
  按钮上写字, 不只放符号: ▶ 在播放器里是「播放」, 拿来当「下一步」会被看错。
-->
<template>
  <div class="step-player">
    <button type="button" aria-label="回到开头" title="回到第一步" @click="stepper.reset()">« 开头</button>
    <button type="button" aria-label="上一步" title="后退一步" @click="stepper.prev()">‹ 上一步</button>
    <button type="button" class="active" :aria-label="stepper.playing.value ? '暂停' : '播放'"
            :title="stepper.playing.value ? '停在当前这一步' : '从当前这一步往后自动播放'" @click="stepper.toggle()">
      {{ stepper.playing.value ? '‖ 暂停' : '▶ 播放' }}
    </button>
    <button type="button" aria-label="下一步" title="前进一步" @click="stepper.next()">下一步 ›</button>
    <input
      type="range" min="0" :max="stepper.total.value - 1" step="1"
      :value="stepper.step.value" aria-label="时间轴" title="拖动跳到任意一步"
      @input="stepper.step.value = Number($event.target.value)"
    />
    <span class="mono step-label">{{ label || `${stepper.step.value + 1} / ${stepper.total.value}` }}</span>
    <select v-model.number="stepper.speed.value" aria-label="播放速度" title="播放速度">
      <option :value="0.5">0.5×</option>
      <option :value="1">1×</option>
      <option :value="2">2×</option>
      <option :value="4">4×</option>
    </select>
  </div>
</template>

<script setup>
defineProps({
  stepper: { type: Object, required: true },
  label: { type: String, default: '' },
})
</script>
