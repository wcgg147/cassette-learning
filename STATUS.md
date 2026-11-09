# 项目状态 / Project Status

> **WIP 快照** —— 本仓库是从主产品运行树中切出、用于开源的**未完成版本**，不是完整产品。
> **WIP snapshot** —— this repo is an *unfinished extract* open-sourced from a larger product, not a finished product.

## 已验证 / Verified
- **E0** 静态：无悬空 import、前端 JS 语法、密钥扫描 0 命中、无凭据文件。
- **E1** 单测：后端 `agent` 536 例 + `test_runner` 431 例全绿（14 个继承红测已逐条审计）。
- **E2** 服务：零 key 起全套，**四盘建盘 / 素材库 / 词块本 / 会话** 全部 200。
- **E3** 浏览器：四盘 + 素材导入（文本→分析→成盘）点击走通，0 前端报错。

## 未完结 / 未验证 / Not done
1. **真机语音对话未验证**：语音代理没配 key 是纯文本模式；ASR 探通过，但回复生成因账号欠费 / key 失效没跑通（环境 / 额度问题，非代码缺陷）。有可用 key 后按 [`docs/ACCEPTANCE.md`](docs/ACCEPTANCE.md) §7 复跑补完。
2. **素材带的 AI 分析依赖 key**：离线时 `hooks / top_chunks / summary` 为空（盘照常建，只是没分析内容）。
3. **两处已知小瑕疵未修**（记在验收档，非阻塞）：场景盘标签误显「闲聊盘」；无 key 下素材带分析为空。

## 相关文档
- 中文说明：[README.md](README.md)（含「当前状态」一节）
- English：[README_EN.md](README_EN.md)
- 抽取清单：[docs/EXTRACTION.md](docs/EXTRACTION.md)
- 验收实测（E0–E3）+ 红测审计：[docs/ACCEPTANCE.md](docs/ACCEPTANCE.md)

许可证 / License：MIT © 2026 WCGG147
