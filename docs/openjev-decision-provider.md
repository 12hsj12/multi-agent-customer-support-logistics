# OpenJev 决策后端接入说明

## 协议边界

EchoMind 调用的是 OpenJev 的 `POST /api/evaluate`，请求固定 `mode: "oneshot"`。OpenJev oneshot 再通过 OpenAI-compatible Chat Completions 接口请求其模型服务，一次请求生成结构化答案。

`POST /v1/systemone` 是另一层协议：它属于 TypeSafe 官方 API，也被 Mapika/decider 兼容服务实现。OpenJev 的 `mode: "decider"` 会把 System One 形状的请求转发到配置的 decider 服务。EchoMind 不把 `/v1/systemone` 误当作 OpenJev 的入口，也不直接加载 Mapika GGUF 来宣称“接入了 OpenJev”。

OpenJev 的 oneshot 请求使用 `state` 和命名 `questions`，choice 候选通过 `criteria` 传递。返回的 `answers.intent.choice` 必须属于 EchoMind 提供的候选集合；否则该请求拒判并保留 baseline 结果。置信值仅作为 OpenJev 的原始信号，不视为与 baseline 已校准。

## Compose 部署

Compose 以固定 Git 提交 `a08e969c37b2e4a37f95b3426f983bd94303590c` 构建 SiliconLabAI/OpenJev，许可证为上游 MIT。OpenJev 仅在 Compose 网络内暴露 3001 端口，没有宿主端口映射；EchoMind 通过 `http://openjev:3001` 调用它。

### 上游 npm 审计记录（2026-10-26）

对该固定提交的构建镜像分别运行 `npm audit --json` 和 `npm audit --omit=dev --json`：本轮两者都报告 3 项 high、0 项 critical，因此 3 项均在生产依赖图中，没有额外的 dev-only high 项。此前 `npm ci` 构建警告曾显示 5 项 high；本轮以锁定镜像的审计 JSON 作为当前记录，并保留该计数差异供后续核查。

| 包与镜像版本 | 审计内容 | 风险判断 |
|---|---|---|
| `body-parser@1.20.8` | GHSA-27r3-jxmh-p828：压缩请求体尾随字节可触发拒绝服务；审计显示从 1.20.9 起有修复版本 | 当前 Express JSON 请求体限制为 2 MB；OpenJev 只在 Compose 私有网络暴露，EchoMind provider 不发送压缩请求。对当前链路不是阻断项，但公开暴露服务前应升级并复测 |
| `zod@3.25.76` | GHSA-mrhx-c7xc-m3w5：校验超大受约束集合时可能造成内存耗尽；审计建议的修复版本为 `4.6.7` | 属运行时依赖；当前请求由应用发送固定意图候选，且服务未发布宿主端口。升级跨越 Zod 主版本，未做未经验证的覆盖 |
| `openai@4.104.0` | 审计项通过上述受影响的 Zod 依赖传递；npm 给出的直接修复建议要求升级至 `openai@7.43.0` | 需要跨多个主版本并重新验证 OpenJev 的 oneshot JSON 响应，当前不盲目升级 |

OpenJev API 本身没有调用方认证，私网边界和当前 EchoMind 固定候选集是这份风险判断的前提。若未来发布 OpenJev 宿主端口、连接不受信任的网络调用方或允许任意 questions/criteria，应先修复并回归测试这些生产依赖告警。

本项目使用 OpenJev 的 oneshot 模式调用 DeepSeek OpenAI-compatible Chat Completions。当前本机 `.env` 将 DeepSeek 凭证命名为 `ANTHROPIC_API_KEY`，其 `ANTHROPIC_BASE_URL` 指向 `/anthropic`，这是 Anthropic-compatible 地址；不能把这个 URL 传给 OpenJev。OpenJev 使用同一 DeepSeek API 凭证，但将密钥映射到独立、被 Git 忽略的 `.env.openjev`，并固定 OpenAI-compatible 地址 `https://api.deepseek.com`。请求体显式包含 `model: "deepseek-flash"`，服务端默认值也设为同一模型。

模型服务凭证单独传给 OpenJev 容器，不传给 EchoMind 主应用：

```powershell
Copy-Item .env.openjev.example .env.openjev
```

本地 `.env.openjev` 只保存 `OPENAI_API_KEY`、`OPENAI_BASE_URL=https://api.deepseek.com` 和 `OPENJEV_MODEL=deepseek-flash`。`.env.openjev` 不纳入版本控制。不要把模型密钥放到 EchoMind 的配置或请求体中；OpenJev 的 `/api/evaluate` 路由本身不提供调用方认证。本项目只把 OpenJev 放在未发布宿主端口的 Compose 私有网络中。

部署选择 `ECHOMIND_DECISION_BACKEND=local` 和 `ECHOMIND_LOCAL_DECISION_PROVIDER=openjev_http`。`OPENJEV_MODE` 默认 `oneshot`，EchoMind 会在 `/api/evaluate` 请求体显式发送 `mode` 和 `model`；设置为 `decider` 可让 OpenJev 转发到 `OPENJEV_DECIDER_BASE_URL` 对应的 `/v1/systemone` 服务。此配置点不会自动部署或要求 GPU 模型服务。默认仍保留现有规则：未失败且 LLM 置信度达到 0.85 时直接采用 baseline；其余只在低融合置信度、`other` 或多路冲突时调用 OpenJev。调用异常、超时、非法候选或拒判均保留 baseline 结果。将 provider 改为 `mapika_gguf` 仍可选择原本地 GGUF 路径。

如果 `.env.openjev` 尚未设置可用模型服务，OpenJev 容器仍能健康启动，但 oneshot 会失败并触发 baseline 回退。成功健康检查只证明 OpenJev HTTP 服务已启动，不代表模型凭证有效或决策请求成功。

## TypeSafe 官方 Jev 预留

`OfficialJevProvider` 按官方协议发送 `model`、`state`、`questions` 到 `/v1/systemone`，并使用 Bearer API Key。默认模型别名是 `jev-latest`。只有显式选择 `ECHOMIND_LOCAL_DECISION_PROVIDER=official_jev` 且设置 `ECHOMIND_OFFICIAL_JEV_API_KEY` 才会构造该 provider。当前无官方 API Key，不会发起官方调用；契约测试只使用 mock 凭证和 mock 响应。

## 验证与配对评测

单元测试：

```powershell
python -m pip install -r requirements-test.txt
python -m pytest tests/test_decision_provider.py
```

Docker E2E 使用真实固定版本的 OpenJev 服务，但把上游 OpenAI-compatible 模型端点替换成确定性本地 stub。它验证 `/api/evaluate` oneshot 请求、`decision_source=openjev` 和非法候选回退；它不测真实模型质量，也不声称接通 TypeSafe 官方服务：

```powershell
docker compose -p echomind-openjev-e2e -f docker-compose.yml -f docker-compose.openjev-e2e.yml run --build --rm echomind-openjev-e2e
```

完整 FastAPI `/chat` E2E 使用真实 DeepSeek + OpenJev 链路，只把 API 发布到本机回环地址。测试覆盖将触发阈值提高到 1.0，只为确定性地触发一次决策；生产默认仍是 `.env.example` 中的阈值。合成复合请求验证 `decision_source=openjev`、应用内 Agent 路由和 `request_id`：

```powershell
docker compose -p echomind-openjev-chat -f docker-compose.yml -f docker-compose.openjev-chat-e2e.yml --profile openjev-chat-e2e run --build --rm echomind-openjev-chat-e2e
```

真实模型配对评测使用 `.env` 中现有的 baseline 配置，以及 `.env.openjev` 中的 OpenAI-compatible DeepSeek 配置。报告记录的 OpenJev 模型应为 `deepseek-flash`：

```powershell
docker compose --profile openjev-benchmark run --build --rm echomind-openjev-benchmark
```

评测使用固定的 `data/eval/g4_intents_v1.json`，运行 baseline 与 OpenJev 两个后端，记录共同旧标签和细分物流标签的 Accuracy / Macro-F1、P50/P95 延迟、OpenJev 调用数、应用决策数、回退状态和逐样本标签。报告新写入 `data/eval/g5_results/`，使用唯一文件名并拒绝覆盖已有文件或 G3 历史报告。若所有调用都失败，报告会将 `valid_model_comparison` 标为 `false`；这种结果只能说明回退行为，不能用于比较模型准确率。

README 的“已接入”功能说明应在实际环境完成 Docker E2E，并至少有一次真实 OpenJev 决策成功后再更新。
