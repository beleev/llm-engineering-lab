<!--
  一行滑杆: 标签 + range + 当前值。 <LabSlider v-model="T" label="温度 T" :min="0.1" :max="5" :step="0.1" />
  传了 format 时, 读屏器读的是 format 之后的文字 (aria-valuetext), 和屏幕上显示的一致。
  拿下标做 v-model 的滑杆 (值是 0 1 2, 显示的是 "64 (= 整张量)") 一定要传 format。
-->
<template>
  <div class="ctl">
    <label :for="id">{{ label }}</label>
    <input
      :id="id" type="range" :min="min" :max="max" :step="step"
      :value="modelValue" :aria-valuetext="format ? `${format(modelValue)}${unit}` : undefined"
      @input="$emit('update:modelValue', Number($event.target.value))"
    />
    <span class="val mono">{{ format ? format(modelValue) : modelValue }}{{ unit }}</span>
  </div>
</template>

<script setup>
import { useId } from 'vue'
defineProps({
  modelValue: { type: Number, required: true },
  label: { type: String, required: true },
  min: { type: Number, default: 0 },
  max: { type: Number, default: 100 },
  step: { type: Number, default: 1 },
  unit: { type: String, default: '' },
  format: { type: Function, default: null },
})
defineEmits(['update:modelValue'])
const id = useId()
</script>
