# MCP：旧握手、新报文与资源端授权

## 直觉

同名协议会修订，旧握手和新请求不能只靠改日期互换。工具权限与 HTTP 凭据校验也分属不同层。

## 核心原理

m09/core 固定 2025-06-18：initialize → initialized。2026-07-28 改为逐请求 _meta 携带版本与能力，HTTP 同时带 MCP-Protocol-Version；server/discover 返回 supportedVersions。未知版本返回 -32022。 HTTP 还镜像 Mcp-Method 与 Mcp-Name；请求头和正文不一致返回 400/-32020，未知版本返回 400/-32022，未知方法返回 404/-32601。

授权对 MCP 实现是可选能力；支持 HTTP 授权时遵循资源端约定。stdio 使用环境凭据，不照搬 HTTP OAuth 流程。本地服务用服务端登记的 opaque token 演示 audience、过期和 scope，未实现登录或令牌签发。

取消与截止时间要进入工具执行器。关闭连接不证明副作用已撤销；本实验在提交点之前检测停止条件。网络断连、幂等与任务恢复参见异步运行时。

## 运行

```bash
python -m llm_agent.m55_mcp_versions.demo
```

需要 Python ≥ 3.10；使用现有 torch 依赖。默认 CPU、本机夹具，无模型 API 调用。

## 运行后应该看到什么

### 2025 stdio

| 读数 | 实际结果 |
| --- | --- |
| HTTP 状态 | "不适用" |
| 协议或授权拒绝 | false |
| 取消/超时后提交 | 0 |


### 2026 HTTP

| 读数 | 实际结果 |
| --- | --- |
| HTTP 状态 | "200" |
| 协议或授权拒绝 | false |
| 取消/超时后提交 | 0 |


### 未知版本

| 读数 | 实际结果 |
| --- | --- |
| HTTP 状态 | "400" |
| 协议或授权拒绝 | true |
| 取消/超时后提交 | 0 |


### 错误受众

| 读数 | 实际结果 |
| --- | --- |
| HTTP 状态 | "401" |
| 协议或授权拒绝 | true |
| 取消/超时后提交 | 0 |


### 凭据过期

| 读数 | 实际结果 |
| --- | --- |
| HTTP 状态 | "401" |
| 协议或授权拒绝 | true |
| 取消/超时后提交 | 0 |


### 权限不足

| 读数 | 实际结果 |
| --- | --- |
| HTTP 状态 | "403" |
| 协议或授权拒绝 | true |
| 取消/超时后提交 | 0 |


### 执行取消

| 读数 | 实际结果 |
| --- | --- |
| HTTP 状态 | "200" |
| 协议或授权拒绝 | false |
| 取消/超时后提交 | 0 |


### 执行超时

| 读数 | 实际结果 |
| --- | --- |
| HTTP 状态 | "200" |
| 协议或授权拒绝 | false |
| 取消/超时后提交 | 0 |


## 与真实系统的差距

HTTP 资源端真实运行于 loopback，固定时钟和 opaque 凭据用于授权反例；未实现 OAuth 登录、令牌签发、SSE 或完整互操作认证。取消/超时检验工具执行层的合作停止，不等于网络断连自动撤销副作用。stdio 的环境凭据与 HTTP 授权流程分开。

## 怎样验证

在正文、实验和 README 同步标明 2025-06-18 教学配置，说明与 2026-07-28 的差异及不支持范围。；增加两个明确版本的报文约定测试，未知版本不得静默按旧报文解析。；可选本地 HTTP 端点覆盖错误 audience、过期凭据、权限不足、取消与超时；明确 OAuth 授权是可选能力，不能把 HTTP 授权流程强加给 stdio。

边界测试见 [test_engineering_completion.py](../../tests/test_engineering_completion.py)。

## 常见误区与自测

把旧 initialize 里的日期改成新协议，是否完成迁移？

<details><summary>展开解析</summary>

取消与截止时间要进入工具执行器。关闭连接不证明副作用已撤销；本实验在提交点之前检测停止条件。网络断连、幂等与任务恢复参见异步运行时。

</details>

## 参考资料

- [MCP 2026-07-28 变更说明](https://modelcontextprotocol.io/specification/2026-07-28/changelog)
- [MCP HTTP Authorization 的适用范围](https://modelcontextprotocol.io/specification/2026-07-28/basic/authorization)

- [MCP Streamable HTTP：请求头校验与状态码](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http)
