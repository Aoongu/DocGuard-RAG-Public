# Ragbits 基础复现说明

## 复现范围

本目录只验证固定上游基线的两条现成功能：本地 Markdown 文档摄取与语义检索，以及上游离线 Chat 示例的 FastAPI 应用构造和公开端点。它没有新增混合检索、重排、拒答、评测算法、Prompt、业务接口或 UI，因而不是 DocGuard-RAG 的二次开发成果。

- 上游：https://github.com/deepsense-ai/ragbits
- 固定上游 commit：`32074a71324239ba677234a67737ed65fb678eeb`
- 许可证：MIT（见仓库根目录 `LICENSE`）
- 复现环境：Windows 11、Python 3.13.5、uv 0.11.13；详见 `environment.md`

## 一键验证

依赖已经按 `environment.md` 安装到仓库根目录的 `.venv` 后，从仓库根目录运行：

```powershell
powershell -ExecutionPolicy Bypass -File reproduction/verify_reproduction.ps1
```

验证脚本统一使用 `uv run --no-sync`，避免 uv 在运行前自动重新求解依赖并改写锁文件。它会运行复现测试和两个脚本、校验 JSON 的 Top-1/HTTP 200 证据、检查固定上游完整跟踪树、限制新增路径白名单，并扫描所有本次新增文本文件中的常见明文 API key 格式。任一步失败都会返回非零退出码。

完整树检查以固定 commit 的全部原始跟踪路径为基准，要求路径集合保留且工作树内容经过 Git clean filter 后与基准 blob 一致；只允许在 `reproduction/**` 和 `docs/superpowers/plans/**` 下新增文件。脚本不硬编码某个最终复现 HEAD，只输出实际运行时 HEAD，因此可继续验证后续仅修改白名单内容的文档提交。

等价的核心命令是：

```powershell
uv run --no-sync pytest reproduction/tests -q
uv run --no-sync python reproduction/run_document_search.py
uv run --no-sync python reproduction/smoke_offline_chat.py
```

## 目录、输入与输出

- `inputs/`：3 份人为编写的最小英文 Markdown 输入，分别描述 Ragbits、内存向量库和 Top-1 指标。
- `run_document_search.py`：只编排上游公开的 `DocumentMeta`、`DocumentSearch`、`FastEmbedEmbedder` 和 `InMemoryVectorStore`。
- `smoke_offline_chat.py`：从上游 `examples.chat.offline_chat:MyChat` 构造 `RagbitsAPI`，用 `TestClient` 请求 `/openapi.json` 和 `/api/config`。
- `tests/`：验证固定查询的预期 Top-1，以及两个 Chat 端点均返回 HTTP 200。
- `results/document_search.json`：三条查询的 Top-1 来源、上游返回的 score 和文本。
- `results/offline_chat_smoke.json`：应用路由数、端点状态码及非敏感响应摘要。
- `results/upstream_tests.txt`、`results/document_search_tests.txt`：前置测试证据。
- `results/verification.txt`：功能复现提交 `39bcb5a67dd73db108708c41a2df4a766302c1c2` 上实际生成的完整验证日志（已脱敏）。它不声称验证包含本说明文件的后续打包文档提交；最终交付 commit 和全包哈希由交付包根目录的 `PACKAGE_VERIFICATION.txt` 外置记录。

`document_search.json` 中的负 score 由上游向量存储的距离/排序约定直接产生；在这套实现中仍按上游排序取第一名。它不是 0 到 1 的相似度概率，不能按概率解释，也不能跨实现直接比较绝对值。

结果稳定性分为两层：JSON schema、预期 Top-1 和 HTTP 200 是必须通过的语义验证；运行前后字节 SHA256 一致只用于检查当前冻结环境的确定性。由于 FastEmbed 模型 revision 未锁定，未来重新下载的同名模型可能改变数值或字节，导致字节检查产生跨时间假失败。遇到这种情况必须先核对模型 revision 与语义结果，不能降低 Top-1/HTTP 断言。

## 环境限制与故障排查

### `uv sync` 在 Windows/Python 3.13 失败

固定 `uv.lock` 包含当前平台无法正常安装的组合：`ray==2.52.1` 没有 Windows wheel；`safetensors==0.4.5` 没有 Python 3.13 的兼容 wheel，而源码构建的包元数据又不能被当前工具链解析。不得为“跑通”而修改 `packages/**`、`examples/**` 或 `uv.lock`。本机采用的最小替代安装命令和完整限制见 `environment.md`。

### FastEmbed 首次运行需要模型缓存

脚本指定模型名 `BAAI/bge-small-en-v1.5`，但未固定模型仓库 revision。当前结果使用本机已有/首次下载的模型文件；未来远端模型内容变化可能影响位级可复现性。交付包不包含模型缓存，离线新机器需自行预置兼容缓存。

### `uv pip check` 返回 3 项不兼容

当前替代环境中实际观察到：

1. `fastapi-cloud-cli==0.26.0` 要求 `typer>=0.26.1`，实际为 `typer==0.19.2`；
2. `ragbits-document-search` 元数据声明依赖 `opencv-python`，当前未安装；
3. `ragbits-document-search` 元数据声明依赖 `rerankers`，当前未安装。

后两项是 `--no-deps` 安装工作区包后留下的元数据依赖缺口。本次最小 Markdown/FastEmbed/InMemory 路径不调用它们，但这不证明其他文档解析、重排或完整安装路径可用。

### Docker 与外部服务

本机没有可用的 Docker 命令，因此 Docker 镜像、Compose、外部向量数据库、GCS、联网 LLM 和云服务均未验证。离线 Chat smoke 只证明应用能够构造且两个公开端点返回 200，不证明对话生成链路或生产部署可用。

## 结果判定

成功验证的精确定义见 `复现结论.md`；来源和文件边界见 `SOURCE_MANIFEST.md`。若模型下载失败、缓存缺失或代理拦截，应先解决 Hugging Face/FastEmbed 模型可用性，再原样重跑验证脚本，不要修改预期 Top-1 或上游源码来掩盖失败。
