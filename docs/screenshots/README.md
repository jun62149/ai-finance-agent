# 截图 / 录屏

放这里的图片会在 README 里引用。建议至少 3 张（PNG，宽度 1200~1600px 足够）：

| 文件名 | 拍什么 | 为什么值得放 |
| --- | --- | --- |
| `01-chat.png` | Web UI 问一个股票问题，展示**报告 + 证据来源**（chunk 标题/来源/发布时间） | 证明"有出处"，不是纯聊天 |
| `02-eval.png` | 检索评测表（四组对照）或 `pytest -q` 的 `218 passed` 输出 | 证明有量化评测与测试 |
| `03-docker.png` | `docker compose ps`（qdrant/api 均 healthy）+ 浏览器打开 `127.0.0.1:8000/` | 证明能一键跑起来 |

在 README 的「效果示例」一节按这个格式引用：

```markdown
![Agent 问答与证据来源](docs/screenshots/01-chat.png)
```

注意：截图里**不要出现 API key**（`.env` 内容、终端里的 `LLM_API_KEY`），也别露出本机用户名路径。
