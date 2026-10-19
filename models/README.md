# 决策模型资源

G4 `local` 评测使用的权重文件：

- 文件名：`decider-2b-v11-Q4_K_M.gguf`
- 发布方：[Mapika/decider-2b-GGUF](https://huggingface.co/Mapika/decider-2b-GGUF)
- 格式：GGUF，Q4_K_M，约 1.3 GB
- 模型许可：Apache-2.0；分发时保留上游许可与来源说明
- SHA-256：`b7c132a67934d51c81abc96bb7724800f965ff5a288aed3e1ca7d8bc349c1386`

本地加载还需要同一上游仓库中的元数据文件，已随项目提供：

- `decider_config.json`
- `tokenizer.json`
- `tokenizer_config.json`
- `chat_template.jinja`
- `LICENSE-APACHE-2.0.txt`（上游 Apache-2.0 许可证文本）

GGUF 权重由 Git LFS 管理；小型配置和模板文件由普通 Git 管理。运行时使用 `llama-cpp-python==0.3.35`、`numpy==1.26.4` 和 `tokenizers==0.20.3`，并通过本目录的 `decide_gguf.py` 读取模型、tokenizer 配置及 Mapika 提示词格式。此适配器保留 GGUF helper 的 choice 推理逻辑，同时避免加载 PyTorch 和 Transformers。

`decide_gguf.py` 基于 [Mapika/decider-2b-GGUF](https://huggingface.co/Mapika/decider-2b-GGUF/blob/main/decide_gguf.py) 及其 `decider.prompt` 适配；适配代码保留提示词格式、选项标签编码和 choice 温度处理，并使用本项目固定的 `tokenizers` 版本读取上游 `tokenizer.json`。上游按 Apache-2.0 许可发布，许可证文本随项目保留。

SiliconLabAI/OpenJev 是决策应用，其 `decider` 模式使用 Mapika Decider 权重。EchoMind 评测的具体文件和版本以本清单及 G4 报告为准；该模型不是官方 JEV 模型。

模型来自 Mapika 的 Apache-2.0 发布；不是官方 JEV 模型。保持上述文件名，以供评测和运行配置引用。
