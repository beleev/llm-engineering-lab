# 文档生命周期：旧答案什么时候必须失效

## 直觉

检索正确不仅是找到相似文本，还要确认此刻有权读、版本仍有效、答案能回到原文。

## 核心原理

HTML 的段落和表格行保留原始字符偏移，chunk 带 source/version/parent 与 embedding 版本。PDF 解析要保留页码和坐标；扫描件还需 OCR 置信度与版面顺序，不能把失败文件当成空成功。

前过滤在合法集合内找候选；后过滤先取 top-k，再删无权项，可能丢掉本应返回的合法项。两者最终返回都要校验权限，原文、引用和缓存也要遵守同一个范围。

更新先完成解析，再替换索引快照；撤权和删除清空缓存。引用读取时再次核对版本及授权，历史引用也会失效。更换编码器不能把不兼容向量混入旧索引。

## 运行

```bash
python -m llm_agent.m54_document_lifecycle.demo
```

需要 Python ≥ 3.10；使用现有 torch 依赖。默认 CPU、本机夹具，无模型 API 调用。

## 运行后应该看到什么

### 先过滤权限

| 读数 | 实际结果 |
| --- | --- |
| 返回片段 | 1 |
| 越权返回 | 0 |
| 旧引用失效 | false |
| embedding 拒绝 | false |
| 当前文档版本 | 1 |


### 后过滤权限

| 读数 | 实际结果 |
| --- | --- |
| 返回片段 | 0 |
| 越权返回 | 0 |
| 旧引用失效 | false |
| embedding 拒绝 | false |
| 当前文档版本 | 1 |


### 文档更新

| 读数 | 实际结果 |
| --- | --- |
| 返回片段 | 1 |
| 越权返回 | 0 |
| 旧引用失效 | true |
| embedding 拒绝 | false |
| 当前文档版本 | 2 |


### 撤销权限

| 读数 | 实际结果 |
| --- | --- |
| 返回片段 | 0 |
| 越权返回 | 0 |
| 旧引用失效 | true |
| embedding 拒绝 | false |
| 当前文档版本 | 1 |


### 删除文档

| 读数 | 实际结果 |
| --- | --- |
| 返回片段 | 0 |
| 越权返回 | 0 |
| 旧引用失效 | true |
| embedding 拒绝 | false |
| 当前文档版本 | 0 |


### 编码器换版

| 读数 | 实际结果 |
| --- | --- |
| 返回片段 | 1 |
| 越权返回 | 0 |
| 旧引用失效 | false |
| embedding 拒绝 | true |
| 当前文档版本 | 1 |


## 与真实系统的差距

两维固定编码器与单桶 ANN 用于隔离权限、召回和失效行为，不代表真实语义效果。HTML 保留字符偏移；PDF/扫描件必须由单独解析/OCR 适配器提供页码与坐标，本模块显式拒绝原始 PDF。缓存按租户与索引代际隔离，变更时保守全量失效。

## 怎样验证

固定带表格的文档，保留 source_id/version/page 或 offset/parent_id；解析失败不得伪造空成功。；两租户分别检索；对比 ANN 前过滤与后过滤的召回影响，任何返回和引用都不得泄漏无权文档。；修改、删除或撤权后重放相同问题，旧 chunk、旧引用和缓存答案均不可继续生效；embedding 版本不兼容时显式重建或拒绝混用。

边界测试见 [test_engineering_completion.py](../../tests/test_engineering_completion.py)。

## 常见误区与自测

权限后过滤没有泄露文档，为什么仍可能找不到答案？

<details><summary>展开解析</summary>

更新先完成解析，再替换索引快照；撤权和删除清空缓存。引用读取时再次核对版本及授权，历史引用也会失效。更换编码器不能把不兼容向量混入旧索引。

</details>

## 参考资料

- [Unstructured：原始文档到结构化元素](https://docs.unstructured.io/open-source/core-functionality/partitioning)
- [Azure AI Search：向量过滤位置会影响召回](https://learn.microsoft.com/en-us/azure/search/vector-search-filters)
