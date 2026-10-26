# DocGuard-RAG / Ragbits 基础复现交付说明

本目录是 Ragbits 固定上游基线的独立基础复现包，归档于外层 DocGuard-RAG 仓库的 `baseline/ragbits-reproduction/`，不是 DocGuard-RAG 二次开发成果。下文相对路径均以该归档包根目录为基准。

## 两级完整性验证

### 第一级：无 Git、无 Python 的包核验

在本目录运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\VERIFY_PACKAGE.ps1
```

该命令只依赖 Windows PowerShell，检查全包 SHA256、固定提交元数据、790 个上游文件的独立树清单、两个结果 JSON 的 schema 与预期 Top-1/HTTP 200、禁止目录和全包文本密钥模式。它不读取 Git 历史，成功输出 `PACKAGE_VERIFY_PASS`；任一失败返回非零退出码。

`SHA256SUMS.txt` 覆盖除自身和外置验证回执 `PACKAGE_VERIFICATION.txt` 以外的全部文件，包括 provenance bundle、bundle 独立 SHA-256 与 Git 级验证器。回执由 `VERIFY_PACKAGE.ps1 -WriteReport` 生成，为避免自引用哈希循环而明确排除；普通核验会检查回执中的固定复现提交、清单哈希与当前包一致。

### 第二级：需要 Git 的来源与模式核验

在外层 DocGuard-RAG Git 工作树中进入本目录后运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\VERIFY_PROVENANCE.ps1
```

该命令验证 `provenance/ragbits-history.bundle` 的独立 SHA-256 和两个固定分支引用，在系统临时目录从 bundle 克隆并执行完整对象检查。它确认 sanitized/squashed 复现快照 `d8ba5d3f00594e0900b7c5c2217935ba0efa2b06` 的唯一 parent 是固定上游提交 `32074a71324239ba677234a67737ed65fb678eeb`，tree 为不可变元数据 `99ec16c9efd9fb85e950a1a9b0f71b004eb736e5`，且私有历史 DAG 不在 bundle 中。随后逐项核对 790 个上游路径的归档文件 Git blob、外层仓库索引 blob 与 mode，并扫描上游到快照间全部新增或修改的普通 blob：可靠识别 UTF-8（有/无 BOM）及带 BOM 的 UTF-16LE/BE，其余对象同时按原始字节的 ASCII、UTF-16LE、UTF-16BE 视图兜底检查敏感模式；未经显式 allowlist 不跳过对象。验证器输出候选、文本解码、原始字节兜底和跳过计数，并实际检出 18 个复现文件。临时克隆在退出前清理；成功输出 `PROVENANCE_VERIFY_PASS`。

## 内容与提交边界

- `upstream_source/`：上游 `deepsense-ai/ragbits` 固定 commit `32074a71324239ba677234a67737ed65fb678eeb` 的 790 个原始 blob 文件及 MIT `LICENSE`，不含 `.git`、`.venv` 或缓存。
- `reproduction/`：复现输入、脚本、测试、结果和说明的 18 个路径均可从 sanitized/squashed 快照 `d8ba5d3f00594e0900b7c5c2217935ba0efa2b06` 检出；快照 tree 与原最终历史提交完全相同。归档副本仅对 `SOURCE_MANIFEST.md` 的 provenance 说明作了包级更新，冻结脚本、输入和结果未改写。
- `plan/`：使用实际 `reproduction/inputs` 路径并与 Top-1 输出一致的实施计划。
- `UPSTREAM_TREE_SHA256.txt`：仅覆盖 `upstream_source/` 的 790 个文件。
- `SHA256SUMS.txt`：交付内容清单。
- `PACKAGE_VERIFICATION.txt`：包清单生成后写出的外置核验回执。
- `provenance/ragbits-history.bundle`：只包含 `refs/heads/main` 与 `refs/heads/repro/ragbits-sanitized-snapshot-20261011` 及其可达历史；不包含原复现分支的 8 个中间提交。
- `provenance/BUNDLE_SHA256.txt`：bundle 的独立 SHA-256。
- `VERIFY_PROVENANCE.ps1`：需要 Git 和外层 DocGuard-RAG 索引的来源、blob 与 mode 核验器。

原最终历史提交 `fc411730eb6bb95eaae02fcc7be5c5be34cceac0` 仅作为快照来源记录；因其可达历史包含早期未脱敏内容，不纳入可分发 bundle。冻结结果文件仍是原复现运行证据，不因 provenance 重写而改写。固定 sanitized 快照和包哈希结果以归档包根目录的 `UPSTREAM_COMMIT.txt`、`PACKAGE_VERIFICATION.txt` 为准，Git 来源可恢复性以 `VERIFY_PROVENANCE.ps1` 为准。

## 功能重跑

功能重跑仍需要兼容的 Python/uv 环境和 FastEmbed 模型。应在完整 Ragbits Git 克隆中检出 `reproduction_commit`，按 `reproduction/environment.md` 建立环境，然后运行：

```powershell
powershell -ExecutionPolicy Bypass -File reproduction/verify_reproduction.ps1
```

纯快照 `upstream_source/` 不含 `.git`，只能用于源码审计或配合复制进去的 `reproduction/` 运行三条 `uv run --no-sync` 核心命令，不能执行仓库历史检查。

语义验证（Top-1/HTTP 200）必须通过。结果 JSON 的字节哈希只证明当前冻结环境确定性；FastEmbed 模型 revision 未锁定，跨时间重新下载可能造成字节检查假失败。Windows/Python 3.13 的 Ray/safetensors 限制、Docker 未验证及 `uv pip check` 三项不兼容详见复现说明。

检索结果中的负 score 是上游距离/排序语义，不是相似度概率。
