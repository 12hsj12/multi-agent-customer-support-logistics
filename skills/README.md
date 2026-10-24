# 动态 Skills 使用说明

服务启动时，ECHOMIND_SKILLS_DIR 指定的目录会被扫描；默认目录为仓库的 skills/。Skill 依据 front matter 中的 enabled、agents 和 keywords 配置匹配用户请求，再按提示预算注入目标 LLM Agent 的 system prompt。修改文件后可调用 POST /skills/reload 热加载。

## 内置 Skills

| 目录 | 适用 Agent | 内容 |
|---|---|---|
| general_customer_service/SKILL.md | general | 接待、澄清、分流、投诉与转人工 |
| technical_support/SKILL.md | technical | 故障排查、错误码、部署与安全边界 |
| billing_support/SKILL.md | billing | 扣款、退款、发票与财务审核边界 |
| logistics_support/SKILL.md | logistics | 物流纠纷分类、信息收集、模拟数据边界与升级候选条件 |

## Skill 文件结构

```text
skills/<skill_name>/SKILL.md
```

示例 front matter：

```markdown
---
name: 物流纠纷处理规范
description: 适用于 LogisticsAgent 的纠纷信息收集与交接规范
keywords: 物流,快递,运单,丢件,破损,签收,延误,地址
agents: logistics
enabled: true
---
```

常用字段：

- name：Skill 名称，会显示在 /skills 状态中。
- description：用途说明。
- keywords：消息关键词，英文逗号或中文逗号分隔。
- agents：适用 Agent 类型。
- enabled：是否启用。

加载器支持 .md、.txt、.json 文件，也支持子目录中的 SKILL.md。单个 Skill 解析失败会被记录，不阻止其他 Skill 加载。总注入长度受 ECHOMIND_SKILLS_MAX_PROMPT_CHARS 限制。

## 查看与热加载

```bash
curl http://localhost:8000/skills
curl -X POST http://localhost:8000/skills/reload
```

物流 Skill 和 data/demo_docs/logistics_demo_policy.md 的职责不同：Skill 规定 Agent 如何收集信息、解释 local_mock 结果以及何时生成应用内交接；模拟文档只用于演示知识库检索，不是承运商政策。
