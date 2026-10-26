# 智能客服多 Agent 协同决策系统

这是一个以 Python / FastAPI 实现的客服 Agent 系统：识别用户诉求后，由 Orchestrator 选择主 Agent 与必要的协作 Agent，Agent 按需调用业务工具和知识检索，最后通过记忆、请求轨迹与监控形成可观察的对话闭环。物流纠纷是系统内的核心业务场景之一，与技术支持、账单及通用客服共享同一套意图、路由、工具和运行时。

## 能力概览

- **细粒度意图识别**：LLM、字符 n-gram 向量相似度和关键词规则三路识别；抽取运单号、订单号、金额、日期和错误码等字段。
- **协同决策与 Agent 路由**：综合意图、领域关键词、实体和 Agent 在线表现选择主路由；复合请求可并行调用多个专业 Agent。
- **可切换的决策后端**：Compose 默认启用独立 OpenJev oneshot 服务，以 DeepSeek `deepseek-flash` 复核不确定结果；保留 baseline 高置信优先和异常回退，Mapika GGUF 仍可选。
- **业务工具**：物流本地模拟查询、技术排障辅助、账单字段检查与金额比较、人工交接摘要，以及 Agent 可按需使用的知识检索。
- **RAG**：Agent 自主决定是否调用共享知识库工具；检索链路包含查询改写、并行召回、合并去重、重排、缓存和故障回退。
- **动态 Skills**：根据 Agent 类型和用户消息匹配业务规则，支持运行时重新加载。
- **分层记忆**：Redis 保存会话工作记忆和摘要；ChromaDB 保存情景记忆、用户画像及知识库向量。
- **监控与评测**：请求级 request_id、工具调用轨迹、Agent/工具运行统计、Prometheus 指标、LLM-as-Judge 和固定意图评测记录。
- **容器部署**：Docker Compose 编排 FastAPI、独立 OpenJev、Redis、ChromaDB、Prometheus 与 Nginx。

## 整体架构

```text
客户端
  -> FastAPI /chat
  -> 读取会话记忆
  -> IntentRecognizer：LLM + 字符 n-gram + Pattern 融合
  -> Compose 默认由 OpenJev oneshot 复核不确定决策（高置信 LLM 优先；失败回退到 baseline）
  -> AgentOrchestrator：主 Agent / 辅助 Agent / 监控降级
  -> Agent Skills + 工具白名单 + 按需工具调用
       ├─ GeneralAgent
       ├─ TechnicalAgent
       ├─ BillingAgent
       ├─ LogisticsAgent -> 本地 mock 物流 fixture
       └─ EscalationAgent -> 应用内交接结果
       └─ 共享 search_knowledge_base -> ChromaDB RAG
  -> 回复、记忆写回、request_id 工具轨迹、Monitor 指标
```

详见[架构图](wiki/架构图.md)、[业务流程说明](wiki/业务流程说明.md)和[完整使用指南](wiki/完整使用指南.md)。

## 意图、路由与协作

物流细分意图包括：

| 意图 | 业务含义 | 路由 |
|---|---|---|
| `LOGISTICS_LOST` | 疑似丢件 | `LogisticsAgent` |
| `LOGISTICS_DAMAGED` | 破损、错发或少件 | `LogisticsAgent` |
| `LOGISTICS_DISPUTE` | 签收记录与实际收货不符 | `LogisticsAgent` |
| `LOGISTICS_DELAY` | 配送延误或轨迹停滞 | `LogisticsAgent` |
| `LOGISTICS_ADDRESS` | 咨询修改收货地址 | `LogisticsAgent` |

普通物流进度咨询仍可走 GeneralAgent。技术类、账单类、通用咨询和人工升级分别由 TechnicalAgent、BillingAgent、GeneralAgent、EscalationAgent 处理。路由器还检查复合领域信号；例如签收争议同时申请退款时，可由 Logistics 与 Billing Agent 并行处理。低置信度 other 可先追问澄清；CRITICAL 或明确转人工意图进入升级路由。

三路 baseline 的融合权重为 LLM 0.7、向量相似度 0.2、关键词模式 0.1。当前 Anthropic SDK 不提供此处使用的 embeddings 资源，向量分支实际回退到本地 256 维字符 n-gram 哈希向量。Compose 默认启用 local 决策，阈值为 0.65；未失败且置信度不低于 0.85 的 LLM 结果优先，不再调用 OpenJev。其他情况下，若融合置信度偏低、结果为 other 或多路结果冲突，才触发 OpenJev `/api/evaluate` oneshot。拒判、超时、缺少模型或其他异常均保留 baseline 结果。OpenJev 的置信信号不视为与 baseline 已校准、可直接比较。Mapika GGUF 可通过 `ECHOMIND_LOCAL_DECISION_PROVIDER=mapika_gguf` 选择。

意图识别的字符 n-gram 哈希向量不是预训练语义 Embedding；ChromaDB 的默认 ONNX Embedding 用于 RAG 知识库和记忆检索，是独立路径。

## 业务工具、RAG 与 Skills

各 Agent 使用白名单工具。物流 query_logistics 只读取仓库内 data/mock_logistics.json，结果标记为 local_mock；服务没有连接真实快递平台，也不会代为修改地址、认定丢件或执行理赔。缺少运单号时先追问。物流交接候选条件可配置：模拟记录停滞天数达到阈值（默认 7 天）、意图为疑似丢件，且最近五条历史消息中至少出现两条同类用户追问。历史消息没有可靠时间戳，因此追问数量只是演示用代理指标。满足规则时系统交给 EscalationAgent 生成应用内交接结果；当前不创建真实工单或通知人工队列。

所有 Agent 均可按需调用 search_knowledge_base。/chat 不会根据意图强制预检索，也不会把检索结果自动注入每个请求。独立的 /search 接口可用于检查查询改写、召回与重排。仓库中的物流政策说明标注为模拟资料，上传后用于演示检索，不构成承运商或法律政策。

Skills 与 RAG 分工不同：知识库提供可检索的业务材料，Skill 提供客服处理边界和步骤。内置 Skill 目录包括通用客服、技术支持、账单服务和物流纠纷；/skills/reload 可在运行时重新加载。

## 记忆、监控与请求轨迹

- **工作记忆**：Redis 保存当前会话消息（TTL 24 小时）；达到 15 条时压缩较早消息，保留最近 5 条。
- **会话摘要**：Redis 保存压缩摘要；ChromaDB 的 episodic collection 保存摘要供语义检索。
- **用户画像与知识**：ChromaDB 的 user_profile 保存画像，knowledge_base 保存 RAG 文档。
- **监控**：/monitor 查看 Agent 和工具统计、告警与路由降权；/metrics 提供 Prometheus 指标。
- **轨迹**：/chat 返回 request_id、识别来源、主辅 Agent、工具和路由信息；/trace/tool/{request_id} 查询该请求的工具调用明细。轨迹缓存在当前服务进程内，默认保留最近 200 条，不是持久化审计库。

## 技术栈

| 层 | 当前实现 |
|---|---|
| API 与模型调用 | Python 3.12 Docker 镜像、FastAPI、Pydantic、Anthropic SDK 兼容接口 |
| Agent 与工具编排 | AgentOrchestrator、角色契约、工具白名单、并行协作、降级路由 |
| 决策后端 | OpenJev `/api/evaluate` oneshot + DeepSeek OpenAI-compatible API、统一 DecisionProvider；Mapika GGUF 可选 |
| 检索与存储 | ChromaDB、Redis、查询改写与重排、ChromaDB 内置向量模型 |
| 监控与评测 | Prometheus Client、在线 Monitor、LLM-as-Judge、固定 JSON benchmark |
| 部署 | Docker Compose、Nginx、Redis、ChromaDB、Prometheus |

项目依赖版本见 [requirements.txt](requirements.txt)，容器 Python 基础镜像见 [Dockerfile](Dockerfile)。

## 固定意图评测记录

下表汇总仓库中同一数据集摘要 e7b9c286a53dadd7df637a6aa91d67d3288aeb33dc796cb156a66111cbb30843 的三份历史报告。样本为合成中文客服文本，不含真实用户数据；结果是已保存记录，本次文档整理没有重跑 benchmark。

| 历史快照 | 共同旧标签准确率 / Macro-F1 | 五类物流细分准确率 / Macro-F1 | 主 Agent 路由 | 意图识别 P95 |
|---|---:|---:|---:|---:|
| baseline | 31/36（86.1%） / 0.9099 | 不支持 | 37/38（97.4%） | 2,214.75 ms |
| 物流扩展（baseline 决策） | 33/36（91.7%） / 0.9186 | 13/15（86.7%） / 0.8714 | 36/38（94.7%） | 2,059.01 ms |
| Mapika local 决策 | 33/36（91.7%） / 0.8788 | 14/15（93.3%） / 0.9314 | 37/38（97.4%） | 13,258.38 ms |

P95 是 38 条单次意图识别耗时的最近秩统计值，不是完整 /chat 的端到端延迟；local 决策器在该报告中实际调用 5 次，local 决策 P95 为 13,303.85 ms。物流细分样本只有 15 条，不能据此推断广泛业务效果。相同准确率也不代表 Macro-F1 相同。

G3 历史报告注明的模型为 Mapika/decider-2b-GGUF:decider-2b-v11-Q4_K_M.gguf，运行记录为 deepseek-flash、decider_ai 1.5.0、llama-cpp-python 0.3.35 和仅作为导入依赖的 torch 2.9.1+cpu。该报告标记 working-tree-G3，其意图识别、Orchestrator 和物流工具源码摘要与当前文件不一致，并记录为评测时直接调用 GGUFDecider.decide，没有调用当前生产 MapikaGgufProvider 路径。报告没有记录操作系统、CPU/GPU、完整 Python 环境或完整源代码提交，因此不能将它描述为当前运行时的可复现性能结果。按当前 benchmark 脚本口径对报告中的 38 条逐样本延迟复算，P95 与保存值吻合；原始执行环境仍不完整。

### OpenJev + DeepSeek 实测

真实 DeepSeek oneshot smoke、OpenJev HTTP Docker E2E 和真实 FastAPI `/chat` 全栈 E2E 均已通过。两次全栈合成请求均返回 `decision_source=openjev`、`decision_status=applied` 和 `request_id`；端到端耗时分别为 16.7/20.5 秒，OpenJev 决策耗时为 6.85/3.66 秒。OpenJev 模型为 `deepseek-flash`。固定 38 条样本两次配对运行均保留：首轮报告 [openjev_paired_20261026T022620262830Z.json](data/eval/g5_results/openjev_paired_20261026T022620262830Z.json)，最终源码报告 [openjev_paired_20261026T024116143097Z.json](data/eval/g5_results/openjev_paired_20261026T024116143097Z.json)。

| 配对运行 | 共同标签准确率 / Macro-F1（baseline → OpenJev） | 五类物流准确率（baseline → OpenJev） | OpenJev 调用 / 采纳 / 回退 |
|---|---|---|---|
| 首轮 | 94.4% / 0.9394 → 91.7% / 0.9368 | 86.7% → 93.3% | 5 / 4 / 1 |
| 最终源码 | 88.9% / 0.9065 → 94.4% / 0.9394 | 86.7% → 100% | 6 / 6 / 0 |

两次运行的整体准确率变化方向不一致，最新一轮又记录到 baseline 与 OpenJev 增强链路各 38 次识别中分别有 4 次、6 次 LLM 解析/API 失败事件；因此不能声称稳定的准确率提升。最终源码运行的 baseline、OpenJev 增强链路端到端 P50/P95 分别为 1,356/2,442 ms、1,513/7,150 ms；OpenJev 决策 P50/P95 为 3,103/5,636 ms。38 条固定样本规模和远程模型的错误/波动限制了结论。详细协议、运行方式和限制见[决策后端说明](docs/openjev-decision-provider.md)。

另有一份较早物流结果 [g2_logistics.json](data/eval/g4_results/g2_logistics.json)，其数据集摘要为 6d4fa7801382ecaf76689dca2b92d2b56aca75d8a1acae07691dd32d6edb1a8a，与上表数据集不同，单独保留为历史记录，不与表中结果混算。所有历史报告和固定样本均保留在 [data/eval/](data/eval/)。

## 启动

从仓库根目录操作。需要 Docker Compose、Git LFS，以及一个 Anthropic 或兼容接口的 LLM API Key。仓库提供 .env.example；它包含占位值，启动前需填入有效 API Key，并替换 Redis 密码。

```bash
git lfs install
git lfs pull
cp .env.example .env
```

编辑 .env，至少设置 baseline 使用的 ANTHROPIC_API_KEY 和唯一的 REDIS_PASSWORD。再将同一 DeepSeek API 凭证安全映射到被 Git 忽略的 `.env.openjev`，设置 `OPENAI_BASE_URL=https://api.deepseek.com` 和 `OPENJEV_MODEL=deepseek-flash`；不要把 Anthropic-compatible 的 `/anthropic` URL 用作 OpenJev 的模型地址。Compose 默认选择 `ECHOMIND_DECISION_BACKEND=local` 与 `ECHOMIND_LOCAL_DECISION_PROVIDER=openjev_http`；若改为 baseline 可禁用决策复核。

```bash
docker compose up -d --build
docker compose ps
docker compose logs -f echomind
```

- API：http://localhost:8000
- Swagger：http://localhost:8000/docs
- Health：http://localhost:8000/health
- Nginx：http://localhost
- ChromaDB / Redis / Prometheus：8001 / 6379 / 9090

Dockerfile 使用 Python 3.12，并在构建时通过 CPU wheel index 安装 llama-cpp-python、下载 ChromaDB ONNX embedding 模型。Docker 配置和依赖需访问相应包源/模型源；镜像构建是否成功取决于这些来源在运行环境中的可访问性。.dockerignore 排除 models/*.gguf，Compose 将本地 models/ 以只读目录挂载到 API 容器，因此权重由 Git LFS 单独提供，不复制进镜像。

如需改用原 Mapika 本地决策器，在 .env 中设置 `ECHOMIND_LOCAL_DECISION_PROVIDER=mapika_gguf`。需先完成 git lfs pull，模型文件位于 models/decider-2b-v11-Q4_K_M.gguf，容器路径由 Compose 设置为 /app/models/decider-2b-v11-Q4_K_M.gguf。本地模型约 1.27 GB。Mapika 模型与 models/decide_gguf.py 的来源和许可证说明见 [models/README.md](models/README.md)；模型采用 Apache-2.0 并保留上游许可证文件。

当前实际使用固定提交构建的 SiliconLabAI/OpenJev 服务，EchoMind 调用其 `/api/evaluate` oneshot 接口。OpenJev 的 `/v1/systemone` decider 模式是转发到兼容服务的另一条路径，本配置未启用；TypeSafe 官方 Jev 适配器只完成 mock 契约测试，未接入官方服务或使用官方 API Key。OpenJev、Mapika 与官方 Jev 是不同实现，不能互相等同。

## 常用接口

| 方法与路径 | 用途 |
|---|---|
| POST /chat | 主对话，返回意图、主辅 Agent、工具、升级状态与 request_id |
| GET /trace/tool/{request_id} | 查询请求级工具调用明细 |
| GET /trace/tools?limit=20 | 查询进程内最近工具轨迹 |
| POST /search | 独立检查知识检索流程 |
| POST /knowledge/add、POST /knowledge/upload | 导入知识文档 |
| GET /knowledge/stats | 查看知识库片段数 |
| GET /skills、POST /skills/reload | 查看与重载 Skills |
| GET /monitor、GET /metrics | 监控摘要与 Prometheus 指标 |
| POST /eval/run | 运行在线意图与回复质量评测 |

POST /eval/run 会更新 EVAL_BASELINE_PATH 指定的回归基线文件。固定 benchmark 脚本默认写入新的 reproduction_<variant>.json 文件；若手动复跑，请先指定新输出路径，避免覆盖任何历史报告。

## 许可证与模型来源

- 本仓库原有许可证与版权文件按仓库内容保留。
- 本地决策模型来自 [Mapika/decider-2b-GGUF](https://huggingface.co/Mapika/decider-2b-GGUF)，Apache-2.0；上游来源、适配说明与许可证文本见 [models/README.md](models/README.md) 与 models/LICENSE-APACHE-2.0.txt。
- 独立 OpenJev 服务固定到 SiliconLabAI/OpenJev 的 MIT 许可提交；它以 DeepSeek 的 OpenAI-compatible API 执行 oneshot 决策，不是 TypeSafe 官方 Jev 服务。官方 Jev 尚未接入。

## 目录

```text
api/                 FastAPI 路由与生命周期
agents/              Agent、路由、协作与业务工具
core/                意图识别、DecisionProvider、Skills 加载
mcp/                 工具调用管理、RAG 知识库
memory/              Redis / ChromaDB 分层记忆
monitor/             在线监控
evaluation/          LLM-as-Judge 评测
scripts/              固定数据集 benchmark
skills/               动态业务规范
models/               Mapika GGUF 适配器、配置与模型资源
data/mock_logistics.json  本地模拟运单 fixture
data/eval/            固定评测集与完整历史结果
wiki/                 架构、流程、使用指南和归档记录
```
