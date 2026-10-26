# data

数据按用途和可追溯性分层：

- `archives/`：已提交、按 SHA-256 固定的 PostgreSQL 官方文档归档，是 M1
  离线恢复的唯一输入根。
- `raw/`：从固定归档恢复的原始 HTML，默认不提交 Git，可随时重建。
- `manifests/`：固定来源、选择规则以及每个受管 HTML 的路径、主题、大小和
  SHA-256。
- `LICENSES/`：上游数据许可证文本及其官方来源。
- `processed/`：清洗、解析和切分后的中间产物，默认不提交 Git，可由原始快照和配置重建。
- `eval/`：经过审核的评测输入与标注，允许提交；冻结测试集必须保持独立，不得用于调参。

任何不可公开或不可再分发的数据都不得进入仓库；生成数据必须能通过 manifest 和脚本追溯到来源。

## M1 PostgreSQL 文档快照

固定版本和官方来源：

| PostgreSQL | 官方归档 | 字节数 | SHA-256 |
| --- | --- | ---: | --- |
| 17.11 | <https://ftp.postgresql.org/pub/source/v17.11/postgresql-17.11-docs.tar.gz> | 3,843,559 | `cd1dbddebbd34ab038f6225a3f66411d1ed6e16b0e7f4b8cfa6f45e99642fa77` |
| 18.6 | <https://ftp.postgresql.org/pub/source/v18.6/postgresql-18.6-docs.tar.gz> | 3,906,887 | `0419dec0d3b7ca55a80c0519a1dd88a8d172019ef9841288889f0281ea1f97ed` |

文档采用 [PostgreSQL License](https://www.postgresql.org/about/licence/)；仓库内副本见
`data/LICENSES/PostgreSQL.txt`。`sources.json` 固定归档元数据和有优先级的主题规则，
`snapshot.jsonl` 固定 773 个 HTML 文件（17.11 为 386 个，18.6 为 387 个）。
归档获取方式、robots 与访问政策核对见 `data/ACCESS_POLICY.md`；该记录不改变归档或
manifest 字节。

六个技术主题按有优先级的白名单规则划分：认证与权限（`auth_privileges`）、备份恢复（`backup_restore`）、性能调优（`performance_tuning`）、安装配置（`installation_configuration`）、SQL 语言（`sql_language`）和 SQL 命令（`sql_commands`）。同一文件只归入首个命中主题；两版各自完整纳入 189 个 `sql-*.html` 页面。

默认恢复完全离线，只读取仓库内已提交归档：

```powershell
uv run python -m docguard.ingestion.snapshot restore
uv run python -m docguard.ingestion.snapshot verify --output-root data/raw
```

只有在归档缺失且明确允许联网时，才使用 `--download-missing`；该选项只访问
`sources.json` 对应 catalog 中固定的 PostgreSQL 官方 URL，并继续强制校验大小与
SHA-256：

```powershell
uv run python -m docguard.ingestion.snapshot restore --download-missing
```

`data/raw/` 是 Git 忽略的可重建目录，不是新的数据源。恢复先在同一输出根创建隐藏随机 staging，完成文件集合、大小和 SHA-256 校验后，再把单个版本目录以不可覆盖的原子操作发布。事务边界是“每个版本原子”，不承诺多个版本全有或全无；若发布前失败，为避免竞态误删外部目录，staging 会保守保留，错误信息会给出需要人工检查的精确路径。S1 不做 HTML 清洗、正文解析、切分、向量化或检索；HTML 解析和切分属于 S2，向量化与检索属于 S3 及后续阶段。

## S2 结构化解析与切分

`s2-chunk-v1` 使用固定的 `intfloat/multilingual-e5-small@614241f` 与
`BAAI/bge-reranker-base@2cfc18c` tokenizer（均为 512 上限），以 384-token Chunk、48-token
目标重叠和 64-token 最小尾块规则处理全部 773 个 S1 页面。正式回执位于
`data/manifests/s2-chunk-v1.json`，可重建输出位于 Git 忽略的
`data/processed/s2-chunk-v1/`。仓库只冻结离线计数所需的 tokenizer 资产，不包含任何模型权重。

当前冻结结果为 7,413 个 Chunk：17.11 为 3,670 个，18.6 为 3,743 个；token 长度
min/p50/p95/max 为 8/267/379/384，空 Chunk 与超限 Chunk 均为 0。六主题分布和确定性哈希见
`stats.json` 与构建回执，30 条分层回查见 `reports/s2_chunk_review.md`。默认 build/verify 完全离线：

```powershell
uv run python -m docguard.processing.pipeline build
uv run python -m docguard.processing.pipeline verify
uv run python -m docguard.processing.pipeline stats
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\verify_s2.ps1
```

processed 数据不提交 Git；JSONL SHA-256、SQLite 逻辑摘要、配置、S1 manifest 和 tokenizer
manifest 的哈希共同构成可复现边界。

## S3 Dense Retrieval 资产与索引

`data/manifests/s3-embedding-model.json` 固定 `intfloat/multilingual-e5-small` 的 commit revision、3 个 ONNX 相关文件、逐文件大小/SHA-256 和总字节数。模型权重保存在 Git 忽略的 `data/models/s3-e5-small/`；首次运行 `scripts/freeze_s3_model.py --download` 需要联网，之后不带 `--download` 只做离线完整性校验。

`configs/retrieval/s3-dense-v1.json` 固定查询/正文前缀、512 输入上限、384 维归一化、Cosine、batch size 32、Top-K=5 以及 S2 chunks 哈希。可重建索引位于 Git 忽略的 `.qdrant/s3-dense-v1/`，正式回执 `data/manifests/s3-dense-v1.json` 固定 7,413 点及版本计数 3,670/3,743；权重和索引均不提交 Git。

`data/eval/s3-smoke-v1.jsonl` 是 10 条功能性 smoke query，不是 S4 黄金评测集。逐题 Top-5 和人工核验分别位于 `reports/s3_dense_smoke_results.jsonl` 与 `reports/s3_dense_smoke_review.md`；本阶段不生成汇总 CSV，不把原始 Cosine score 解释为概率。
