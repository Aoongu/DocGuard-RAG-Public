# S3 Dense Retrieval Smoke 人工核验

## 核验边界

- 数据：冻结的 PostgreSQL 17.11 与 18.6 技术文档，共 7,413 个 `s2-chunk-v1` chunk。
- 检索：`intfloat/multilingual-e5-small` 固定 revision，384 维、归一化、Cosine、Top-5。
- 方法：逐题检查 Top-5 的版本、主题、章节、正文与冻结的 PostgreSQL 原文链接。
- 判定：`pass` 表示结果可直接解释问题；`partial` 表示主要结果正确但 Top-5 中存在较弱结果；`fail` 表示主要结果不能解释问题。
- 本文件只记录 smoke 观察，不计算整体质量指标，也不据此修改模型、查询或参数。score 是原始 Cosine 排序分数，不是概率或置信度。

## 逐题记录

| Query ID | 版本过滤 | 判定 | Top-1 原文 | 观察 |
|---|---|---|---|---|
| `s3q01` | 全部 | pass | [Primary Keys](https://www.postgresql.org/docs/17/ddl-constraints.html#DDL-CONSTRAINTS-PRIMARY-KEYS) | Top-1 直接给出 `CREATE TABLE ... PRIMARY KEY`、唯一且非空约束及自动创建 B-tree 索引；Top-5 均围绕主键语法或语义。 |
| `s3q02` | 全部 | partial | [MVCC Introduction](https://www.postgresql.org/docs/18/mvcc-intro.html#MVCC-INTRO) | 前两项分别来自 18.6/17.11，直接解释 MVCC 快照与读写不互斥；后续结果主要是 TRUNCATE/COPY 的 MVCC 注意事项，相关但不能独立完整回答定义。 |
| `s3q03` | 17.11 | pass | [VACUUM](https://www.postgresql.org/docs/17/sql-vacuum.html#id-1.9.3.184.5) | Top-1 直接说明回收 dead tuples、普通 VACUUM 重用空间与 VACUUM FULL 归还操作系统空间；全部结果均为 17.11，后续项覆盖 autovacuum 和参数。 |
| `s3q04` | 17.11 | pass | [B-Tree](https://www.postgresql.org/docs/17/indexes-types.html#INDEXES-TYPES-BTREE) | Top-1 为 B-tree 类型说明，后续结果覆盖 ORDER BY、多列索引、CREATE INDEX 示例，且无版本泄漏。 |
| `s3q05` | 17.11 | pass | [START TRANSACTION](https://www.postgresql.org/docs/17/sql-start-transaction.html#synopsis) | Top-1/2 的 synopsis 明确列出 Serializable、Repeatable Read、Read Committed、Read Uncommitted；其余结果补充设置方式和隔离注意事项。 |
| `s3q06` | 18.6 | partial | [Replication Settings](https://www.postgresql.org/docs/18/runtime-config-replication.html#RUNTIME-CONFIG-REPLICATION) | Top-1、3、4、5 覆盖 streaming replication 的全局、primary、sender、standby 配置；Top-2 是 `pg_basebackup` 传输速率选项，属于部署相关但不是核心配置答案。 |
| `s3q07` | 18.6 | pass | [EXPLAIN ANALYZE](https://www.postgresql.org/docs/18/using-explain.html#USING-EXPLAIN-ANALYZE) | Top-1 至 Top-4 均为 EXPLAIN ANALYZE 章节，能够解释实际执行、时间和行数信息；全部结果均为 18.6。 |
| `s3q08` | 18.6 | pass | [Row Security Policies](https://www.postgresql.org/docs/18/ddl-rowsecurity.html#DDL-ROWSECURITY) | Top-1 直接解释启用 RLS、默认拒绝、策略表达式与命令/角色范围；Top-5 均属于策略定义、执行或安全注意事项，且无版本泄漏。 |
| `s3q09` | 全部 | pass | [jsonb Indexing](https://www.postgresql.org/docs/17/datatype-json.html#JSON-INDEXING) | Top-5 全部来自 `jsonb Indexing`，包含 GIN、`jsonb_ops`、`jsonb_path_ops`、可索引操作符与查询示例。 |
| `s3q10` | 全部 | pass | [Database Roles](https://www.postgresql.org/docs/17/user-manag.html#USER-MANAG) | Top-5 覆盖角色概念、对象所有权、成员关系、连接身份、CREATEDB/CREATEROLE 属性，能支撑角色与权限管理的概览回答。 |

## 保留观察

- 未指定版本时，相同或近似文档可能同时返回 17.11 与 18.6，这是当前跨版本检索合同的预期行为。
- 指定版本的 `s3q03`–`s3q08` 未观察到跨版本结果。
- `s3q02` 与 `s3q06` 暴露了纯 Dense Top-5 的语义扩散，但按实验协议保留原结果，留待 S4 使用冻结评测集统一量化。
