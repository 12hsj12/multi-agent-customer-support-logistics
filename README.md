# EchoMind

EchoMind 是一个面向客服/运营场景的多 Agent 智能系统。它不是单纯的聊天机器人，而是把以下能力串成闭环：

- 细粒度意图识别
- 路由驱动的多 Agent 编排
- 意图驱动 RAG 检索
- Redis + ChromaDB 分层记忆
- 动态 Skills 注入
- 在线监控与路由降权
- LLM-as-Judge 端到端评测

## 项目文档

- [架构图](wiki/架构图.md)
- [完整使用指南](wiki/完整使用指南.md)
- [业务流程说明](wiki/业务流程说明.md)
- [文档中心](wiki/文档中心.md)

## 快速开始

### 1. 准备环境

- Docker
- Docker Compose
- `ANTHROPIC_API_KEY`

如果使用兼容 Anthropic 协议的第三方模型服务，也可以配置：

```env
ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic
ANTHROPIC_MODEL=deepseek-v4-pro
ANTHROPIC_API_KEY=your_key
```

### 2. 配置环境变量

复制示例配置：

```bash
cp .env.example .env
```

最少确认这些变量可用：

```env
ANTHROPIC_API_KEY=your_api_key
REDIS_PASSWORD=replace-with-a-unique-random-password
```

启动前请把 `REDIS_PASSWORD` 改成唯一随机值。Compose 会在密码缺失时停止启动；Redis、ChromaDB 和 Prometheus 的主机端口只绑定到本机。

### 3. 启动服务

推荐直接启动全栈：

```bash
docker compose up -d --build
```

查看状态：

```bash
docker compose ps
```

看日志：

```bash
docker compose logs -f echomind
```

### 4. 访问入口

- API: `http://localhost:8000`
- Swagger: `http://localhost:8000/docs`
- Nginx: `http://localhost`
- Health: `http://localhost:8000/health`

## 核心功能

### 对话主链路

`POST /chat`

流程是：

```text
读取记忆 -> 意图识别 -> 知识检索 -> Agent 路由 -> 回复生成 -> 写回记忆
```

### 知识库

- `POST /search`
- `POST /knowledge/add`
- `POST /knowledge/upload`
- `GET /knowledge/stats`

### Skills

- `GET /skills`
- `POST /skills/reload`

### 监控与评测

- `GET /monitor`
- `POST /eval/run`

#### 固定意图评测记录（G4）

以下为同一固定数据集 `echomind-g4-intents-v1`（SHA-256：`e7b9c286a53dadd7df637a6aa91d67d3288aeb33dc796cb156a66111cbb30843`）的既有报告。旧标签集有 36 条、细分物流标签有 15 条、Agent 路由有 38 条；本次 G5 集成没有重跑整套评测。

| 版本 | 旧标签准确率 | 物流细分准确率 | Agent 路由准确率 | 意图 P95 |
|---|---:|---:|---:|---:|
| G0 原版 baseline | 31/36（86.1%） | 不支持 | 37/38（97.4%） | 2.21 秒 |
| G2 物流扩展 | 33/36（91.7%） | 13/15（86.7%） | 36/38（94.7%） | 2.06 秒 |
| G3 local 决策增强 | 33/36（91.7%） | 14/15（93.3%） | 37/38（97.4%） | 13.26 秒 |

G3 的细分物流样本量较小，且 local 意图 P95 明显高于 baseline；完整指标与失败案例见 [G0 报告](data/eval/g4_results/g0_baseline.json)、[G2 报告](data/eval/g4_results/g2_logistics_final.json)、[G3 报告](data/eval/g4_results/g3_local.json)。

## 项目结构

```text
api/main.py                  FastAPI 入口
agents/agent_orchestrator.py 多 Agent 编排、物流 Agent 与路由
core/intent_recognizer.py    三路融合与可选决策复核
core/decision_provider.py    决策接口与 OpenJev 兼容适配
core/skill_loader.py         动态 Skills 加载
memory/conversation_memory.py  Redis + ChromaDB 记忆
mcp/tool_manager.py          工具层、缓存、熔断、重排
mcp/knowledge_base.py        ChromaDB 知识库
monitor/performance_monitor.py 在线监控
evaluation/evaluator.py      端到端评测
wiki/                       详细文档
skills/                     动态业务规则
data/                       持久化数据与 mock fixture
data/eval/g4_intents_v1.json 固定 G4 评测集
data/eval/g4_results/       三种版本的 G4 结果
models/                     决策模型资源
```

## 运行时架构

```text
用户请求
  -> /chat
  -> MemoryManager 读取工作记忆、情景记忆、用户画像
  -> IntentRecognizer 三路合并输出初步意图
  -> local 模式仅在低置信、other 或结果冲突时调用决策后端
  -> 决策失败时保留三路识别结果
  -> 输出最终 intent / intent_group / urgency / entities
  -> 按意图决定是否检索知识库
  -> AgentOrchestrator 路由到 Logistics / General / Technical / Billing / Escalation
  -> Skills 注入、工具调用、回复生成
  -> 写回 Redis 和 ChromaDB
  -> Monitor 采集在线指标
  -> Evaluator 做意图识别和回复质量评测
```

## 主要端口

| 服务 | 端口 |
|---|---:|
| EchoMind API | 8000 |
| ChromaDB | 8001 |
| Redis | 6379 |
| Prometheus | 9090 |
| Nginx | 80 |

## 开发和调试

常用顺序：

```text
1. /health
2. /chat
3. /skills
4. /monitor
5. /eval/run
```

## 一句话概括

EchoMind 是一个可观测、可评测、可降级的多 Agent 客服运行时。

## 意图决策后端

默认使用原三路意图识别（`baseline`）。设置 `ECHOMIND_DECISION_BACKEND=local` 后，EchoMind 会在同一 Python 进程内加载 Mapika Decider GGUF；Docker Compose 会把 `models/` 只读挂载到 API 容器，不会把 1.27 GB 权重打进镜像。

```env
ECHOMIND_DECISION_BACKEND=local
ECHOMIND_LOCAL_DECISION_PROVIDER=mapika_gguf
ECHOMIND_DECIDER_MODEL_PATH=/app/models/decider-2b-v11-Q4_K_M.gguf
ECHOMIND_DECIDER_N_CTX=2048
ECHOMIND_DECIDER_N_THREADS=6
ECHOMIND_DECISION_TRIGGER_CONFIDENCE=0.65
ECHOMIND_LLM_PRIORITY_CONFIDENCE=0.85
```

启用 `local` 前先安装 Git LFS 并拉取权重；普通 Git 文件包含 tokenizer 和模型配置：

```bash
git lfs install
git lfs pull
```

模型首次在有冲突、低置信或 `other` 的请求上懒加载，推理在线程池执行并串行访问同一模型实例。只有 LLM 没有高置信结论时才请求；加载失败、拒判或输出无效时保留原三路识别结果。模型置信度作为自身信号记录，不与原三路分数直接比较。`/chat` 响应带有 `intent_decision_source`、`intent_decision_status` 和 `intent_decision_latency_ms`。

GGUF 推理沿用 Mapika GGUF helper 的提示词格式和模型温度配置，通过项目内 `GGUFDecider` 适配器加载权重。运行依赖固定为 `llama-cpp-python==0.3.35`、`numpy==1.26.4` 和与 ChromaDB 兼容的 `tokenizers==0.20.3`；这样无需在 API 环境安装 PyTorch 和 Transformers。Docker 构建通过 `llama-cpp-python` CPU wheel index 安装运行库。若直接在本机安装 Python 依赖，也使用同一 index：

```bash
python -m pip install --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu -r requirements.txt
```

`ECHOMIND_LOCAL_DECISION_PROVIDER=openjev_http` 可保留现有 OpenJev 兼容 HTTP 适配器，需另行配置服务地址。Mapika 开源模型不是官方 JEV；未来接入官方 JEV 时，在同一 `DecisionProvider` 协议下新增官方适配器，不预设未核验的官方 API 格式。模型文件、来源、许可与校验值见 [models/README.md](models/README.md)。
