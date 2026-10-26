# Ragbits 基础复现环境

## 固定基线

- 复现日期：2026-10-11（Asia/Shanghai）
- 上游仓库：https://github.com/deepsense-ai/ragbits
- 分支：`repro/ragbits-baseline-20261011`
- 固定 commit：`32074a71324239ba677234a67737ed65fb678eeb`
- 操作系统：Microsoft Windows 11 家庭版 中文版 10.0.26200（Build 26200）
- Python：3.13.5（由本机 Python 解释器创建 `.venv`；保存日志中的解释器绝对路径已脱敏）
- uv：0.11.13（4512a3931 2026-05-10，x86_64-pc-windows-msvc）
- Docker：未安装或不在 PATH；`docker --version` 返回“无法识别 docker”。

## 核心包版本

```text
ragbits-core==1.7.0.dev202605130309
ragbits-document-search==1.7.0.dev202605130309
ragbits-chat==1.7.0.dev202605130309
ragbits-agents==1.7.0.dev202605130309
fastembed==0.9.0
pytest==8.3.5
pytest-asyncio==0.24.0
docling==2.65.0
datasets==3.6.0
fastapi==0.143.0
pydantic==2.14.0
sqlalchemy==2.1.4
tzdata==2026.5
```

工作区包版本来自固定 commit 的包元数据；由于锁文件在 Windows/Python 3.13 上包含不可安装项，第三方依赖的替代安装由 `uv pip` 重新解析，并非完全锁定环境。该差异是本次复现的主要限制。

## 实际安装命令与结果

计划原命令：

```powershell
uv sync --frozen --package ragbits-document-search --group dev
uv sync --frozen --package ragbits-chat --group dev
uv pip install "fastembed>=0.5.0,<1.0.0"
```

前两条均失败：在指定 `--package` 后，uv 0.11.13 在对应包级项目中查找 `dependency-groups.dev`，而两个包仅声明了 `project.optional-dependencies`，报错为 `Group dev is not defined in the project's dependency-groups table`。第三条成功，安装 `fastembed==0.9.0`。

随后验证了合并的无开发组同步：

```powershell
uv sync --frozen --package ragbits-document-search --package ragbits-chat --no-default-groups
```

该命令仍失败。锁文件选择 `safetensors==0.4.5`，Windows/Python 3.13 没有兼容 wheel；源码构建又因其 `pyproject.toml` 缺少可解析的 `project.version` 而失败。未修改上游源码或锁文件来掩盖问题。

实际采用的最小替代安装命令：

```powershell
uv pip install "fastembed>=0.5.0,<1.0.0" "pytest~=8.3.3" "pytest-asyncio~=0.24.0" "pytest-cov~=5.0.0"
uv pip install -e packages/ragbits-core -e packages/ragbits-agents -e packages/ragbits-chat
uv pip install --no-deps -e packages/ragbits-document-search
uv pip install "sqlalchemy[asyncio]>=2.0.39,<3.0.0" "aiosqlite>=0.21.0" "greenlet>=3.1.1"
uv pip install "docling==2.65.0" "datasets>=3.0.1,<4.0.0" tzdata
```

上述命令只补齐本次离线测试直接需要的工作区包、文档解析依赖、SQL 测试依赖和 Windows IANA 时区数据；没有安装 Ray、GCS、Unstructured 或外部数据库/云服务依赖。

## 测试结论

计划原测试命令：

```powershell
uv run pytest packages/ragbits-core/tests/unit/vector_stores/test_in_memory.py packages/ragbits-document-search/tests/unit/test_document_search.py packages/ragbits-chat/tests/unit -q
```

该命令在执行 pytest 前触发 uv 自动同步，并因 `ray==2.52.1` 没有 Windows wheel 而失败。改用 `uv run --no-sync` 后：

- 补齐直接测试依赖后的完整集合：`148 passed, 7 failed, 43 warnings`。
- 剩余 7 项：1 项 Windows 已打开临时文件权限差异，3 项测试用 `/` 拆分路径导致 Windows 路径断言失败，3 项 GCS 可选依赖未安装。
- 排除上述 7 个已解释失败后，保留完整 Chat 单测目录的离线基线：`148 passed, 7 deselected, 38 warnings`，退出码 0。

最终通过命令：

```powershell
uv run --no-sync pytest packages/ragbits-core/tests/unit/vector_stores/test_in_memory.py packages/ragbits-document-search/tests/unit/test_document_search.py packages/ragbits-chat/tests/unit -q -k "not test_document_search_ingest_from_source and not test_document_search_ingest_from_uri_with_wildcard and not test_document_search_ingest_from_gcs_uri"
```

该集合覆盖 `InMemoryVectorStore`、DocumentSearch 的字面量/本地文档摄取与检索主流程，以及 `packages/ragbits-chat/tests/unit` 下的完整 Chat 单元测试。详细原始命令、错误和摘要见 `reproduction/results/upstream_tests.txt`。

## 完整冻结清单

实际 `.venv` 的完整包清单保存在 `reproduction/results/pip_freeze.txt`。该文件来自 `uv pip freeze`，为保护机器路径，四个 editable 工作区包的绝对 URL 已改写为仓库根目录下的相对路径。复核当前环境可运行：

```powershell
uv pip freeze
```

该文件首先用于审计和比较实际环境。若需要在同一 commit、同一平台上精确回放已安装版本，可从仓库根目录运行：

```powershell
uv pip install --no-deps -r reproduction/results/pip_freeze.txt
```

`--no-deps` 是必要的：这是实际环境快照，而不是经过统一依赖求解生成的新锁文件；直接重新求解会暴露替代安装环境中的 `fastapi-cloud-cli==0.26.0` 与 `typer==0.19.2` 约束冲突。回放前应创建独立虚拟环境，回放后必须重新运行本页测试命令。该清单不等同于上游 `uv.lock` 的跨平台锁定结果；editable 包要求固定 commit 的仓库源码存在，且 Ray/GCS/Unstructured/Docker 等未验证能力仍受前述平台和可选依赖约束。

## 日志脱敏

保存的测试日志保留了实际命令、错误类型、依赖链和测试计数，但将本机用户名、用户缓存绝对路径、Python 安装绝对路径及测试运行时随机生成的临时密钥替换为占位符或概括性说明。脱敏不改变退出码、失败原因或测试摘要。

## 未验证项

- Ray 分布式摄取（上游锁定版本不支持 Windows）。
- GCS、Unstructured、外部向量数据库及其他可选后端。
- Docker 服务集成（本机无 Docker 命令）。
- Linux/macOS 上的完整锁文件同步。
- 联网 LLM、云模型、远程数据库和端到端在线调用。
- 计划完整测试集合中的 7 个 Windows/可选依赖相关失败未被声明为通过。

## 源码边界

本任务证据文件为 `reproduction/environment.md`、`reproduction/results/upstream_tests.txt` 与 `reproduction/results/pip_freeze.txt`。验证命令 `git diff 32074a71324239ba677234a67737ed65fb678eeb -- packages` 无输出，说明 `packages/**` 未修改。
