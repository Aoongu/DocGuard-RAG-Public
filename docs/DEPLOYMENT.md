# 容器部署与运行

## 前提与配置

使用 Docker Engine 和 Docker Compose。复制 `.env.example` 为 `.env`，至少配置 `DEEPSEEK_API_KEY`（生成与Agent决策）和 `SERPAPI_API_KEY`（网页搜索）；搜索密钥与大模型密钥不同。PowerShell 启动脚本也会读取已有 Windows 用户环境中的同名密钥。

文档下载可设置 `DOCGUARD_DOCUMENT_DNS=doh` 和 `DOCGUARD_DOH_PROVIDER=auto`。HTTPS 始终校验证书、检查公网地址和 robots，网络失败采用有限重试和备用公网 IP；同站点连续失败后暂停。

## 启动

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File ./scripts/start_docker.ps1
```

也可以运行通用命令：

```sh
docker compose config --quiet
docker compose up -d --build
docker compose ps
docker compose logs -f bootstrap api
```

访问 http://127.0.0.1:8080。Nginx 将 `/api/` 转发给内部 API，并关闭 SSE 缓冲、支持长请求。宿主不需要另外启动 Vite 或 Uvicorn。

## 资产与初始化

镜像包含代码和固定配置，不包含模型、密钥、运行文档或索引。完整检索要求已按 README 准备 `data/models/`、`data/processed/`、`.bm25/` 和 `.qdrant/`；这些目录通过挂载提供。

`bootstrap` 是一次性初始化服务，完成后 API 才启动：

1. 对已有 `.qdrant/s3-dense-v1/qdrant` 使用 SQLite 一致性备份，再把现成向量和 payload 写入服务端 Qdrant；不重新下载模型或计算向量。
2. 从 `reports/generated/s15-discovery/knowledge/` 复制已发布的活动知识库快照，校验清单文件哈希；不复制临时目录和任务日志。
3. 重复启动跳过已初始化数据；不覆盖容器里已存在的知识库或不属于初始化工具的向量集合。修复网络或源资产后可重启初始化步骤。

首次没有资产时初始化会显式报告缺失，API 能以降级状态启动，缺失能力不可用于完整问答。新导入文档走独立知识库索引，不会改写 PostgreSQL 冻结语料。

## 持久化与操作

- `qdrant_data`：服务端向量存储。
- `runtime_reports`：任务账本、导入知识库、运行报告和初始化回执。
- `data/`、`.bm25/`：宿主资产；`.qdrant/` 以只读方式提供初始向量。

```sh
docker compose logs -f api
docker compose logs bootstrap
docker compose up -d --build
docker compose down
```

普通 `down` 保留数据卷。`down --volumes` 会删除容器数据，不能作为日常重启命令。源知识库更新不会自动覆盖容器内同名库；容器内继续导入/管理即可。

## 验证与边界

`/api/v1/health` 返回服务依赖状态；`/api/v1/discovery/capabilities` 返回搜索/模型配置是否齐备。健康HTTP请求成功不表示模型、索引或外部API已经全部验证。

冻结评测执行锁与零自动重试规则保持不变。网络重试只用于文档下载，不会重跑冻结 Test 的失败结果。Docker运行不自动发起付费生成或搜索请求。


## 镜像与依赖下载排错

应用文档的DoH设置不会改变Docker引擎的镜像仓库解析。若构建在拉取基础镜像时失败，先检查Docker访问镜像仓库的能力；可从可信缓存源预拉取并核对镜像摘要，再使用原标签构建。依赖安装使用锁文件，并配置下载超时与重试；代码更新会复用独立的依赖构建层。

当前服务端固定为 Qdrant 1.19.1，与客户端版本对齐。向量卷使用版本化名称 `*_qdrant_data_v119`；原旧版向量卷保留，初始化从本地冻结索引复制向量。任务与导入知识库仍保存在原 `runtime_reports` 卷。
