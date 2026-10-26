# S2 Chunk 分层抽查记录

## 抽样规则

- 输入：正式 `s2-chunk-v1` 的 7,413 条 Chunk。
- 分层：PostgreSQL 17.11/18.6 × 六个主题，共 12 个 strata。
- 数量：每个主题固定 5 条，并在两个版本间按 3/2、2/3 交替分配，两个版本各 15 条。
- 排序：每个 stratum 按 `SHA-256(chunk_id)` 升序取前 N 条，不进行人工挑样。
- 核查：逐条回查 S1 原始 HTML 与解析后的同 anchor block，检查标题路径、来源页、正文词汇、
  token 上限以及代码/表格上下文。30 条 Chunk 的正文词汇均来自对应来源 block，无额外生成内容。
- URL 说明：原文没有显式章节 ID 时，`anchor` 是稳定 fallback；此时链接仍定位到 canonical 原文页，
  并以标题路径完成页内回查。

## 抽查结果

| # | chunk_id | 版本 | 主题 | 标题路径 | tokens | 代码 | 表格 | 来源 | 结论 |
| ---: | --- | --- | --- | --- | ---: | --- | --- | --- | --- |
| 1 | `chk_4ceac00455c15000f682acb30c21dcfb6941322a44467e28fb75528c32714eb5` | 17.11 | `auth_privileges` | Synopsis | 88 | 是 | 否 | [原文页](https://www.postgresql.org/docs/17/sql-altergroup.html#synopsis) | pass |
| 2 | `chk_a95f76c296a1178a5d2d604ca83cfc7cb1b10f394b903ae7a49d7e82cbc43352` | 17.11 | `auth_privileges` | 20.15. Authentication Problems | 308 | 是 | 否 | [原文页](https://www.postgresql.org/docs/17/client-authentication-problems.html#CLIENT-AUTHENTICATION-PROBLEMS) | pass |
| 3 | `chk_fb2592e038685a49c8c9e1cac22b37ba134729952afe6b1151598858cdbbe1d4` | 17.11 | `auth_privileges` | 20.3. Authentication Methods | 315 | 否 | 否 | [原文页](https://www.postgresql.org/docs/17/auth-methods.html#AUTH-METHODS) | pass |
| 4 | `chk_816d82ea4f28c92ec9e65d1192290c4f25cb4a398839bb33996fdeade71687cd` | 18.6 | `auth_privileges` | Notes | 28 | 否 | 否 | [原文页](https://www.postgresql.org/docs/18/sql-grant.html#SQL-GRANT-NOTES) | pass |
| 5 | `chk_f5f171b1b5c14c11eeed1840994c7a565c0dc8e4d6133161c58abe1db8bd87d8` | 18.6 | `auth_privileges` | Description | 356 | 否 | 否 | [原文页](https://www.postgresql.org/docs/18/sql-createpolicy.html#id-1.9.3.75.5) | pass |
| 6 | `chk_249e80d45a70bc9df1992bab752b47876509499f9ade14551fa3b6a5696d8f14` | 17.11 | `backup_restore` | Description | 300 | 否 | 否 | [原文页](https://www.postgresql.org/docs/17/app-pgbasebackup.html#id-1.9.4.10.5) | pass |
| 7 | `chk_beac9d206dccf264062d3789a4818c68d34eeb901b81c863cc1a92a078dfa9ee` | 17.11 | `backup_restore` | Options | 297 | 否 | 否 | [原文页](https://www.postgresql.org/docs/17/app-pgdump.html#PG-DUMP-OPTIONS) | pass |
| 8 | `chk_877339545ff4026ff6fccdca8c633f1e68bc3498828f4518fa6647514f97421a` | 18.6 | `backup_restore` | pg_restore | 32 | 否 | 否 | [原文页](https://www.postgresql.org/docs/18/app-pgrestore.html#APP-PGRESTORE) | pass |
| 9 | `chk_a07c91746864f969e96211f0a95f633317ed2c9570ad05376a3a50540430cf04` | 18.6 | `backup_restore` | 25.3. Continuous Archiving and Point-in-Time Recovery (PITR) > 25.3.5. Recovering Using a Continuous Archive Backup | 379 | 否 | 否 | [原文页](https://www.postgresql.org/docs/18/continuous-archiving.html#BACKUP-PITR-RECOVERY) | pass |
| 10 | `chk_ca4b60385580fb3befe46d4192188ae9a3e43766eb3f71e69e728d9cdeeec4eb` | 18.6 | `backup_restore` | Description | 372 | 否 | 否 | [原文页](https://www.postgresql.org/docs/18/app-pgreceivewal.html#id-1.9.4.17.5) | pass |
| 11 | `chk_a4308f510b46c854eeff4299215489e36c83000bf284dfcb43bd8d8fe49a1d99` | 17.11 | `installation_configuration` | 19.11. Client Connection Defaults > 19.11.1. Statement Behavior | 361 | 否 | 否 | [原文页](https://www.postgresql.org/docs/17/runtime-config-client.html#RUNTIME-CONFIG-CLIENT-STATEMENT) | pass |
| 12 | `chk_c0848bfb864db256debd54f54894d5145e14a6275e99f9a2d6ea10e615bb5ef9` | 17.11 | `installation_configuration` | 17.3. Building and Installation with Autoconf and Make > 17.3.3. configure Options > 17.3.3.5. Miscellaneous | 207 | 否 | 否 | [原文页](https://www.postgresql.org/docs/17/install-make.html#CONFIGURE-OPTIONS-MISC) | pass |
| 13 | `chk_d1ae953503a3b3504c39de1885e889223f44ff6dde55dbbdd09e3b9146cdc794` | 17.11 | `installation_configuration` | 19.8. Error Reporting and Logging > 19.8.1. Where to Log | 311 | 否 | 否 | [原文页](https://www.postgresql.org/docs/17/runtime-config-logging.html#RUNTIME-CONFIG-LOGGING-WHERE) | pass |
| 14 | `chk_b4be320f873746fb742b962d6ac5ece6d51cd32dc8d62fa99a038d3a1196253e` | 18.6 | `installation_configuration` | 19.5. Write Ahead Log > 19.5.6. Recovery Target | 348 | 否 | 否 | [原文页](https://www.postgresql.org/docs/18/runtime-config-wal.html#RUNTIME-CONFIG-WAL-RECOVERY-TARGET) | pass |
| 15 | `chk_d894521e454a01e39d37d24280ec3ff62e6729d7fb42513be887ec5b7d8459ce` | 18.6 | `installation_configuration` | 18.2. Creating a Database Cluster | 132 | 否 | 否 | [原文页](https://www.postgresql.org/docs/18/creating-cluster.html#CREATING-CLUSTER) | pass |
| 16 | `chk_72e6e94d9f93c677da426d6c4f8597635cc3387d4d813c32f9a8cad1c258bab7` | 17.11 | `performance_tuning` | 19.4. Resource Consumption > 19.4.1. Memory | 369 | 否 | 否 | [原文页](https://www.postgresql.org/docs/17/runtime-config-resource.html#RUNTIME-CONFIG-RESOURCE-MEMORY) | pass |
| 17 | `chk_bcd7d78497e9ae5b0c57369c1231324c8549c38ea8ba43c389ae9f5021dc21f9` | 17.11 | `performance_tuning` | 11.8. Partial Indexes | 384 | 是 | 否 | [原文页](https://www.postgresql.org/docs/17/indexes-partial.html#INDEXES-PARTIAL) | pass |
| 18 | `chk_6182a264507116640af2c927c1ef9a65a45c74679d73b50e1b28092a6f42e941` | 18.6 | `performance_tuning` | 14.1. Using EXPLAIN > 14.1.2. EXPLAIN ANALYZE | 345 | 是 | 否 | [原文页](https://www.postgresql.org/docs/18/using-explain.html#USING-EXPLAIN-ANALYZE) | pass |
| 19 | `chk_c407f36d467a34368778356ab4def7239597443591a7cc30a902a6d716294f64` | 18.6 | `performance_tuning` | 27.2. The Cumulative Statistics System > 27.2.13. pg_stat_io | 374 | 否 | 是 | [原文页](https://www.postgresql.org/docs/18/monitoring-stats.html#MONITORING-PG-STAT-IO-VIEW) | pass |
| 20 | `chk_ce17c0ab0e144de2671f0451f86bfcd36a64c40bb57994af03b8a6bfeef537c5` | 18.6 | `performance_tuning` | 27.2. The Cumulative Statistics System > 27.2.3. pg_stat_activity | 367 | 否 | 是 | [原文页](https://www.postgresql.org/docs/18/monitoring-stats.html#MONITORING-PG-STAT-ACTIVITY-VIEW) | pass |
| 21 | `chk_21f0afd09c992953387377c10998000d498fa26c090f710b03362e3b7099ae21` | 17.11 | `sql_commands` | Notes | 368 | 否 | 否 | [原文页](https://www.postgresql.org/docs/17/sql-createtrigger.html#SQL-CREATETRIGGER-NOTES) | pass |
| 22 | `chk_534abd5b4044a9c16855d83bfef81e5871653319eebbb045181ec189d1522b1e` | 17.11 | `sql_commands` | File Formats > Binary Format > File Trailer | 85 | 否 | 否 | [原文页](https://www.postgresql.org/docs/17/sql-copy.html#id-1.9.3.55.9.4.7) | pass |
| 23 | `chk_599db67cf50c74af0bcd041a8153d6108db694d8c9deb9c16777c109d4687e74` | 17.11 | `sql_commands` | CREATE TRIGGER | 17 | 否 | 否 | [原文页](https://www.postgresql.org/docs/17/sql-createtrigger.html#SQL-CREATETRIGGER) | pass |
| 24 | `chk_7fb12ed11f2c4913ead441b66f5b1fa0f219f456d72b05bead6a650bedc439b4` | 18.6 | `sql_commands` | Parameters > ORDER BY Clause | 356 | 否 | 否 | [原文页](https://www.postgresql.org/docs/18/sql-select.html#SQL-ORDERBY) | pass |
| 25 | `chk_e7af23460784c6adb836543372c292133bd43d95aff61aa8a595d861399bdef9` | 18.6 | `sql_commands` | Examples | 140 | 是 | 否 | [原文页](https://www.postgresql.org/docs/18/sql-alterview.html#id-1.9.3.45.8) | pass |
| 26 | `chk_2ab77b1b7b9d1f9b229b633f06229e0eac9d3fccb56ff3c4230e7b942a9ad287` | 17.11 | `sql_language` | 5.12. Table Partitioning > 5.12.3. Partitioning Using Inheritance > 5.12.3.1. Example | 381 | 是 | 否 | [原文页](https://www.postgresql.org/docs/17/ddl-partitioning.html#DDL-PARTITIONING-INHERITANCE-EXAMPLE) | pass |
| 27 | `chk_3e3a8d3a27e04ecc5b27cc78ee23a852ab6606e48a064fe883879abb1d2d35cd` | 17.11 | `sql_language` | 5.12. Table Partitioning > 5.12.5. Partitioning and Constraint Exclusion | 353 | 否 | 否 | [原文页](https://www.postgresql.org/docs/17/ddl-partitioning.html#DDL-PARTITIONING-CONSTRAINT-EXCLUSION) | pass |
| 28 | `chk_aee4d369f0451dac9d897791ea5ceb0357217473b2a446f040988933b7c855b7` | 18.6 | `sql_language` | 8.5. Date/Time Types > 8.5.2. Date/Time Output | 384 | 否 | 是 | [原文页](https://www.postgresql.org/docs/18/datatype-datetime.html#DATATYPE-DATETIME-OUTPUT) | pass |
| 29 | `chk_c418adfaef3de97600275919d6523b39ec76f566894abb12c324ec9844e5b449` | 18.6 | `sql_language` | 4.1. Lexical Structure > 4.1.2. Constants | 84 | 否 | 否 | [原文页](https://www.postgresql.org/docs/18/sql-syntax-lexical.html#SQL-SYNTAX-CONSTANTS) | pass |
| 30 | `chk_e5bf9c8397d326ed1d3110ac1d8f73a711d74310a5ed298c2dd1011fc3e85e49` | 18.6 | `sql_language` | 9.21. Aggregate Functions | 227 | 否 | 是 | [原文页](https://www.postgresql.org/docs/18/functions-aggregate.html#FUNCTIONS-AGGREGATE) | pass |

## 结论

30 条分层样本全部为 `pass`。样本覆盖两个版本、六个主题、普通段落、Synopsis/SQL 代码、配置示例、
系统视图表格和 384-token 边界；未发现标题路径污染、来源错配、空 Chunk、超限、代码脱离上下文或
表格缺失表头。首次抽查发现并修复了标题永久链接 `#` 污染和嵌套 DocBook 章节 ID 继承问题，
本表记录的是修复后重新构建、重新抽样的最终结果。
