# MCP: 旧握手、新报文与资源端授权

## 直觉

MCP (Model Context Protocol) 是模型应用调用外部工具时用的一套通信约定, 版本用日期标记。这组实验让同一个客户端用两个版本连工具服务: 2025-06-18 版通过 stdio (起一个子进程, 用它的标准输入输出收发消息) 连接, 先发 initialize、再发 initialized 完成一次握手, 然后用 tools/list 列出 2 个工具; 2026-07-28 版改走 HTTP, 每条请求都在正文的 _meta 字段和 HTTP 头里各写一遍版本号。把版本号改成服务端不认识的 1900-01-01, 得到 HTTP 400 和错误码 -32022; 凭据的受众 (audience, 这张凭据是给哪个服务用的) 不对或已过期回 401, 凭据缺少需要的权限范围 (scope) 回 403; 工具执行被取消或超时的两种情形, 实际提交的副作用都是 0 笔。协议版本、HTTP 凭据和工具执行是三层各自独立的检查, 只把日期改新换不来迁移。

## 核心原理

两个版本定下协议版本的方式不同。仓库里 m09 和 core 用的是 2025-06-18: 连接建立时客户端发 initialize, 服务端回应后客户端再发 initialized, 版本在这一次握手里定下, 之后的消息都默认沿用。2026-07-28 不再依赖握手: 每条请求都在 _meta 字段里带上版本和客户端能力, 走 HTTP 时还要在 MCP-Protocol-Version 头里再写一遍; 客户端可以先调 server/discover, 查服务端支持哪些版本 (supportedVersions)。服务端用错误码区分问题: 头和正文里的版本不一致回 400 和 -32020, 版本不认识回 400 和 -32022, 方法名不存在回 404 和 -32601。HTTP 头里还要镜像正文里的方法名 (Mcp-Method) 和工具名 (Mcp-Name), 两处不一致同样按 -32020 拒绝。

授权在 MCP 里是可选能力, 两种连接方式各管各的。stdio 是本机子进程, 用环境里已有的凭据, 不照搬 HTTP 那套 OAuth (一种标准的授权签发流程)。支持 HTTP 授权时, 由资源端 (提供工具的服务) 检查每个请求带的 token: 本例用的是 opaque token (客户端看不懂内容的一串字符), 服务端拿它去自己登记的表里查出受众、过期时间和 scope。受众不是 lab-mcp 或已经过期 (服务端时钟固定在 100, 这张凭据 50 就过期了) 回 401, scope 里没有 tools:read 回 403。本例没有实现登录和令牌签发, 只演示这三种拒绝。

取消和截止时间必须一直传到真正执行工具的那一层。客户端断开连接, 不代表工具已经做了的事 (例如写数据库、扣款) 会自动撤销。本例的执行器在提交副作用之前检查两个停止条件: 是否已被取消, 当前时间 100 是否已经到了截止时间 99; 任一成立就不提交, 所以取消和超时两种情形提交都是 0 笔。网络断连、幂等 (同一个操作重复执行只生效一次) 和任务恢复放在异步运行时那一章讲。

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

固定同一个工具服务和同一条 tools/list 请求 (列出可用工具), 比较八种情形: 2025 stdio、2026 HTTP、未知版本、错误受众、凭据过期、权限不足、执行取消、执行超时。判定看两件事: 每种情形回的 HTTP 状态码和错误码是否和预期一致, 以及取消和超时之后实际提交的副作用是否为 0 笔。

边界测试见 [test_engineering_completion.py](../../tests/test_engineering_completion.py)。

## 常见误区与自测

把旧 initialize 里的日期改成新协议的日期, 算完成迁移了吗?

<details><summary>展开解析</summary>

不算。旧版本只在 initialize 握手里说一次版本, 之后每条消息都默认对方没变; 2026-07-28 换了一整套报文约定, 日期只是其中一项。第一, 版本要写进每一条请求的 _meta 字段。第二, 走 HTTP 时 MCP-Protocol-Version 头里要再写一份, 而且必须和正文一致, 否则回 400 和 -32020。第三, 服务端支持哪些版本、有哪些能力, 靠 server/discover 查询, 不再靠一次握手协商。实验里版本号不认识就直接回 400 和 -32022, 并在错误里列出支持的版本, 服务端不会按旧报文去猜你的意思。

</details>

## 参考资料

- [MCP 2026-07-28 变更说明](https://modelcontextprotocol.io/specification/2026-07-28/changelog)
- [MCP HTTP Authorization 的适用范围](https://modelcontextprotocol.io/specification/2026-07-28/basic/authorization)

- [MCP Streamable HTTP: 请求头校验与状态码](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http)
