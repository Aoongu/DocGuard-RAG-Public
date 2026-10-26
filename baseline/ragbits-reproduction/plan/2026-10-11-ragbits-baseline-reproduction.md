# Ragbits 基础复现 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在固定上游提交 `32074a71324239ba677234a67737ed65fb678eeb` 上复现 Ragbits 的基础文档检索与离线 Chat 能力，不增加任何检索、重排、拒答或评测算法。

**Architecture:** 保持 `packages/` 上游源码不变，仅在 `reproduction/` 中增加复现输入、最小运行脚本、测试和原始结果。使用上游 `DocumentSearch`、`FastEmbedEmbedder`、`InMemoryVectorStore` 与离线 Chat 示例；所有结论以实际命令日志为准。

**Tech Stack:** Python 3.13、uv、Ragbits 1.7.0.dev202605130309、FastEmbed、pytest、FastAPI；Windows PowerShell。

---

### Task 1: 固定环境并验证上游基线

**Files:**
- Create: `reproduction/environment.md`
- Create: `reproduction/results/upstream_tests.txt`
- Do not modify: `packages/**`

- [ ] **Step 1: 同步最小上游包与测试依赖**

Run:

```powershell
uv sync --frozen --package ragbits-document-search --group dev
uv sync --frozen --package ragbits-chat --group dev
uv pip install "fastembed>=0.5.0,<1.0.0"
```

Expected: `.venv` 创建成功，`uv run python -c "import ragbits"` 退出码为 0。

- [ ] **Step 2: 运行离线、无外部服务的上游单元测试**

Run:

```powershell
uv run pytest packages/ragbits-core/tests/unit/vector_stores/test_in_memory.py packages/ragbits-document-search/tests/unit/test_document_search.py packages/ragbits-chat/tests/unit -q
```

Expected: 退出码 0；若上游测试路径或依赖导致失败，记录原始错误并缩小到等价的离线测试集合，不修改上游实现掩盖失败。

- [ ] **Step 3: 记录环境与测试证据**

`environment.md` 必须记录 OS、Python、uv、上游 commit、包版本、Docker 缺失状态、实际安装命令和未验证项；`upstream_tests.txt` 保存完整测试摘要。

- [ ] **Step 4: 提交**

```powershell
git add reproduction/environment.md reproduction/results/upstream_tests.txt
git commit -m "docs: record ragbits reproduction environment"
```

### Task 2: 复现本地文档摄取与语义检索

**Files:**
- Create: `reproduction/inputs/ragbits_overview.md`
- Create: `reproduction/inputs/vector_store.md`
- Create: `reproduction/inputs/evaluation.md`
- Create: `reproduction/run_document_search.py`
- Create: `reproduction/tests/test_reproduction.py`
- Create: `reproduction/results/document_search.json`

- [ ] **Step 1: 写失败测试**

测试必须调用复现脚本的异步入口，断言三条固定查询各自的 Top-1 来源分别命中文档 `ragbits_overview.md`、`vector_store.md`、`evaluation.md`，并验证结果包含查询、来源、分数和文本。

- [ ] **Step 2: 运行测试并确认先失败**

Run:

```powershell
uv run pytest reproduction/tests/test_reproduction.py -v
```

Expected: 因 `run_document_search.py` 尚不存在或接口未实现而失败。

- [ ] **Step 3: 实现最小复现脚本**

脚本只能编排上游公开类：`DocumentMeta`、`DocumentSearch`、`FastEmbedEmbedder("BAAI/bge-small-en-v1.5")`、`InMemoryVectorStore`。不得新增混合检索、重排、拒答、Prompt 或自定义评分算法。固定查询运行后把逐条 Top-1 结果写入 `reproduction/results/document_search.json`。

- [ ] **Step 4: 运行测试与脚本**

Run:

```powershell
uv run pytest reproduction/tests/test_reproduction.py -v
uv run python reproduction/run_document_search.py
```

Expected: 测试通过；JSON 生成且三条查询 Top-1 命中对应文档。

- [ ] **Step 5: 提交**

```powershell
git add reproduction/inputs reproduction/run_document_search.py reproduction/tests reproduction/results/document_search.json
git commit -m "test: reproduce ragbits local document search"
```

### Task 3: 复现上游离线 Chat API

**Files:**
- Create: `reproduction/smoke_offline_chat.py`
- Create: `reproduction/results/offline_chat_smoke.json`

- [ ] **Step 1: 写 API smoke harness**

从上游 `examples.chat.offline_chat:MyChat` 构造 Ragbits API，使用 FastAPI `TestClient` 检查 OpenAPI 或可用健康/配置端点。不得修改上游示例。

- [ ] **Step 2: 运行 smoke harness**

Run:

```powershell
uv run python reproduction/smoke_offline_chat.py
```

Expected: 退出码 0；JSON 记录实际命中端点、HTTP 状态与响应摘要。若上游无公开健康端点，至少验证应用构造和 `/openapi.json` 返回 200。

- [ ] **Step 3: 提交**

```powershell
git add reproduction/smoke_offline_chat.py reproduction/results/offline_chat_smoke.json
git commit -m "test: reproduce ragbits offline chat api"
```

### Task 4: 形成独立可交付复现包

**Files:**
- Create: `reproduction/README_复现说明.md`
- Create: `reproduction/复现结论.md`
- Create: `reproduction/SOURCE_MANIFEST.md`
- Create: `reproduction/verify_reproduction.ps1`

- [ ] **Step 1: 编写说明与边界**

README 必须包含一键命令、目录、环境、输入、输出和故障排查；结论必须区分“已验证”“未验证”“不是二次开发”；manifest 必须记录上游 URL、commit、MIT 许可证和本次新增文件。

- [ ] **Step 2: 编写验证脚本**

脚本运行两个 reproduction 测试/脚本，检查两个 JSON、上游 commit 的完整原始跟踪树及新增路径白名单，扫描全部新增文本文件中的密钥模式，任一失败返回非零退出码。

- [ ] **Step 3: 执行完整验证**

Run:

```powershell
powershell -ExecutionPolicy Bypass -File reproduction/verify_reproduction.ps1
git diff 32074a71324239ba677234a67737ed65fb678eeb -- packages
git status --short
```

Expected: 验证脚本退出码 0；`packages/**` 无差异；仅有计划内 reproduction/docs 文件和提交。

- [ ] **Step 4: 提交**

```powershell
git add reproduction docs/superpowers/plans/2026-10-11-ragbits-baseline-reproduction.md
git commit -m "docs: package ragbits baseline reproduction"
```

- [ ] **Step 5: 复制用户交付物**

将固定提交的上游源码（排除 `.git`、`.venv`、缓存）与 `reproduction/`、计划、许可证复制到 `outputs/DocGuard-RAG_Ragbits基础复现_20261011/`，并生成文件 SHA256 清单。
