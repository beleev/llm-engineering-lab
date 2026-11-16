# MCP: 旧握手、新报文与资源端授权

## 直觉

同一个客户端先走 2025-06-18 的 stdio 握手, 子进程里 initialize → initialized → tools/list 找到 2 个工具; 再走 2026-07-28 的 HTTP 端点, 版本放进每条请求的 _meta 和 header。把版本改成 1900-01-01 得到 400 / -32022; 凭据受众错、过期回 401, 缺 scope 回 403; 取消和超时两种情形提交 0 笔。协议版本、HTTP 凭据和工具执行是三层不同的检查, 改个日期换不来迁移。

## 核心原理

m09/core 固定 2025-06-18: initialize → initialized 一次握手定版本。2026-07-28 改为每条请求在 _meta 里带版本与能力, HTTP 再带 MCP-Protocol-Version 头, server/discover 返回 supportedVersions; 请求头和正文不一致回 400/-32020, 未知版本回 400/-32022, 未知方法回 404/-32601, HTTP 还镜像 Mcp-Method 与 Mcp-Name。

授权对 MCP 实现是可选能力; 支持 HTTP 授权时按资源端的约定走, stdio 用环境凭据, 不照搬 HTTP 的 OAuth 流程。本地服务用服务端登记的 opaque token 演示 audience、过期和 scope 三种拒绝 (401、401、403), 没有实现登录或令牌签发。

取消与截止时间要进到工具执行器里: 关闭连接不证明副作用已撤销, 本实验在提交点之前检测停止条件, 两种情形都提交 0 笔。网络断连、幂等与任务恢复放在异步运行时一章。

## 运行

```bash
python -m llm_agent.m55_mcp_versions.demo
```

需要 Python ≥ 3.10, 只用标准库和仓库里的 `llm_agent.core.mcp`。HTTP 端点起在 127.0.0.1 的随机端口上, 不出本机; 「2025 stdio」一项会拉起 m09 的 MCP server 子进程。不调用模型 API。

## 运行后应该看到什么

除「2025 stdio」外, 7 种情况都向同一个 loopback 端点发 tools/list, 协议版本写在 _meta 和 MCP-Protocol-Version 头里。「取消/超时后提交」8 行全是 0。

### 2025 stdio

| 读数 | 实际结果 |
| --- | --- |
| HTTP 状态 | "不适用" |
| 协议或授权拒绝 | false |
| 取消/超时后提交 | 0 |

真实子进程 stdio: initialize → initialized → tools/list。协议固定 2025-06-18, 发现 2 个工具。没有 HTTP, 状态写「不适用」。

### 2026 HTTP

| 读数 | 实际结果 |
| --- | --- |
| HTTP 状态 | "200" |
| 协议或授权拒绝 | false |
| 取消/超时后提交 | 0 |

tools/list 返回 200; 接着 server/discover 返回 supportedVersions, tools/call echo 把 "hello" 原样送回。

### 未知版本

| 读数 | 实际结果 |
| --- | --- |
| HTTP 状态 | "400" |
| 协议或授权拒绝 | true |
| 取消/超时后提交 | 0 |

版本写成 1900-01-01: 400, 错误码 -32022, data 里列出 supported 只有 2026-07-28。服务端不会把它按旧报文静默解析。

### 错误受众

| 读数 | 实际结果 |
| --- | --- |
| HTTP 状态 | "401" |
| 协议或授权拒绝 | true |
| 取消/超时后提交 | 0 |

token 的 audience 是 other-api, 不是本服务的 lab-mcp: 401, invalid_token, 响应头带 WWW-Authenticate。

### 凭据过期

| 读数 | 实际结果 |
| --- | --- |
| HTTP 状态 | "401" |
| 协议或授权拒绝 | true |
| 取消/超时后提交 | 0 |

token 的 expires=50, 服务端时钟固定在 100: 401, invalid_token。audience 和过期在服务端是同一条判断, 客户端自报的 claims 不算数。

### 权限不足

| 读数 | 实际结果 |
| --- | --- |
| HTTP 状态 | "403" |
| 协议或授权拒绝 | true |
| 取消/超时后提交 | 0 |

token 合法但 scopes 为空, 没有 tools:read: 403, insufficient_scope。凭据校验过了, 工具权限这一层没过。

### 执行取消

| 读数 | 实际结果 |
| --- | --- |
| HTTP 状态 | "200" |
| 协议或授权拒绝 | false |
| 取消/超时后提交 | 0 |

HTTP 200 之后进工具执行器, cancelled 标志已置位, run() 在提交前返回 cancelled, committed 仍是 0。

### 执行超时

| 读数 | 实际结果 |
| --- | --- |
| HTTP 状态 | "200" |
| 协议或授权拒绝 | false |
| 取消/超时后提交 | 0 |

deadline=99, now=100, run() 返回 deadline, 同样 0 笔提交。

## 与真实系统的差距

HTTP 资源端真的跑在 loopback 上, 但时钟固定在 100, 凭据是服务端写死的 opaque token, 只够做授权反例。没有 OAuth 登录、令牌签发、SSE, 也没有完整的互操作认证。

取消 / 超时检验的是工具执行层的合作停止, 不等于网络断了副作用就自动撤销。stdio 的环境凭据和 HTTP 授权流程是分开的两套。

## 怎样验证

固定同一个工具服务和同一条 tools/list 请求, 比较 2025 stdio、2026 HTTP、未知版本、错误受众、凭据过期、权限不足、取消、超时八种情形; 判定看 HTTP 状态码和错误码是否符合预期, 以及取消/超时后提交是否为 0。

边界测试见 [test_engineering_completion.py](../../tests/test_engineering_completion.py)。

## 常见误区与自测

把旧 initialize 里的日期改成新协议的日期, 算完成迁移了吗?

<details><summary>展开解析</summary>

不算。2026-07-28 不只是换日期: 版本要进每条请求的 _meta, header 要再带一份且和正文一致, 能力靠 server/discover 协商, 不再靠一次握手。实验里版本不认识就回 400 / -32022, 服务端不会按旧报文猜。

</details>

## 参考资料

- [MCP 2026-07-28 变更说明](https://modelcontextprotocol.io/specification/2026-07-28/changelog)
- [MCP HTTP Authorization 的适用范围](https://modelcontextprotocol.io/specification/2026-07-28/basic/authorization)

- [MCP Streamable HTTP: 请求头校验与状态码](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http)
