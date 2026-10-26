# PostgreSQL 文档归档访问政策记录

检查日期：2026-10-14（Asia/Shanghai）

## 获取方式

S1 不进行网页抓取，也不遍历 PostgreSQL 在线文档链接。项目只在显式使用
`--download-missing` 时，以固定 User-Agent 直接下载两个已经确定的官方发布归档：

- <https://ftp.postgresql.org/pub/source/v17.11/postgresql-17.11-docs.tar.gz>
- <https://ftp.postgresql.org/pub/source/v18.6/postgresql-18.6-docs.tar.gz>

默认恢复和全部验收均读取仓库内已提交的归档，不访问网络。下载器使用有限重试、有限超时、
断点续传和固定大小/SHA-256 校验，不进行高频请求或并发爬取。

## robots 与政策检查

- <https://www.postgresql.org/robots.txt> 在检查日可访问；其中限制 `/admin/`、
  `/account/`、`/docs/devel/`、`/list/`、`/search/` 和部分 message-id 路径，
  未列出上述固定发布归档。该文件属于 `www.postgresql.org`，不直接支配
  `ftp.postgresql.org` 主机。
- <https://ftp.postgresql.org/robots.txt> 在检查日返回 `404 Not Found`，未发布
  该主机专用 robots 规则。两个固定归档 URL 的 HEAD 请求均返回 `200 OK`，并公开
  `Content-Length`、`Last-Modified`、`ETag` 和 `Accept-Ranges`。
- PostgreSQL 的政策索引 <https://www.postgresql.org/about/policies/> 未列出针对官方发布归档的
  额外自动下载政策。S1 因为不抓取 HTML 页面，网页爬虫路径规则对归档获取记为“不适用”；
  仍保留本记录以证明获取方式经过人工核对。

## 许可与复核触发条件

文档复制与分发遵循 PostgreSQL License，仓库副本位于
`data/LICENSES/PostgreSQL.txt`。若未来改变来源主机、改为抓取网页、增加并发、扩大版本范围，
或上游 robots/政策发生变化，必须重新检查并更新本记录；本次检查不改变两份归档、
`sources.json` 或 `snapshot.jsonl` 的任何字节。
