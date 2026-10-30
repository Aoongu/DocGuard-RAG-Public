> 当前 Docker 启动和文档导入初始化方式见 [DEPLOYMENT.md](DEPLOYMENT.md)。以下保留历史工程门禁说明。

# S11 本地运行、Docker 与回归门禁

## 安全边界

S11 不解锁冻结 Test，不自动调用 DeepSeek，也不把 API Key、Embedding/Reranker 权重、原始语料或运行期索引写入镜像。镜像只有代码、冻结配置、公开小型 fixture 与依赖；运行制品通过宿主目录或 named volume 提供。

`compose.yaml` 启动 API、前端和 Qdrant。缺少模型、索引或密钥时，API 仍能启动，健康接口返回 `degraded`，查询失败关闭为 `UPSTREAM_ERROR`。这代表“最小演示可启动”，不等于完整问答资产已经准备好。

## Windows 本地模式

要求 PowerShell、Git、uv、Python 3.11 和 Node.js 22：

推荐直接运行一键脚本。它会完成冻结依赖同步、前端测试与构建，另开 PowerShell 启动 Vite，在当前窗口启动 API，并在两个服务就绪后打开浏览器：

```powershell
# 在仓库根目录执行
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\start_local.ps1
```

需要停止服务时，在当前窗口按 `Ctrl+C` 停止 API，并关闭脚本打开的前端 PowerShell 窗口。

等价的手动命令如下：

```powershell
# 在仓库根目录执行
$env:UV_CACHE_DIR = Join-Path (Get-Location) '.uv-cache'
uv sync --frozen --group dev
Push-Location frontend
npm ci
npm test -- --run
npm run build
Pop-Location
uv run docguard-api
```

另开 PowerShell，在 `frontend/` 运行 `npm run dev -- --host 127.0.0.1`，访问 `http://127.0.0.1:5173`。完整问答前仍需按主 README 准备 S2、Dense、BM25、Embedding、Reranker 与 `DEEPSEEK_API_KEY`；密钥只设置在当前进程环境。

重建 API 会后台依次执行 S2、Dense、BM25 三段冻结构建。相同进程内重复提交会返回同一个运行中 job ID；状态保存在 `reports/generated/s11-rebuild/`。在线批量评测只允许 Development/Validation 的 B0/B1/M1/M2；B2/M4 在 S12 正式运行协议前明确拒绝，防止误花费或改写冻结结论。

## Docker 最小演示

要求 Docker Engine 与 Compose v2：

```powershell
# 在仓库根目录执行
docker compose config --quiet
docker compose build api frontend
docker compose up -d
docker compose ps
```

访问 `http://127.0.0.1:8080`。`docker compose down` 后再次 `up -d` 不删除 Qdrant 的 named volume `qdrant_data`；只有显式 `docker compose down --volumes` 才会删除，执行前必须确认。

Compose 将宿主 `data/`、`.bm25/` 与 named volume `runtime_reports` 挂载到 API；新版本先由 bootstrap 初始化已有向量与知识库，详细步骤见 DEPLOYMENT.md。运行期 job 数据不会写回只读镜像。`DOCGUARD_QDRANT_URL=http://qdrant:6333` 让查询走服务模式。生产问答前需把与 `s3-dense-v1` 相同 collection 和 payload 写入服务；S11 CI 使用固定向量实际校验 local/service Top-K 一致性，不会把模型权重放进镜像。

## Qdrant 一致性规则

- Top-K `chunk_id` 集合完全一致。
- 同一 `chunk_id` 的分数差不超过 `1e-6`。
- 只有分数差不超过 `1e-6` 的并列组允许内部顺序不同。
- 非并列项换位、缺失、重复或非有限分数均失败。

本地 Qdrant 已监听 6333 时执行：

```powershell
$env:S11_QDRANT_URL = 'http://127.0.0.1:6333'
uv run pytest -q tests/integration/test_s11_qdrant_parity.py
```

## 回归与 CI

固定小型 fixture 不加载真实模型，门槛为 Recall@5 ≥ 0.50、MRR@10 ≥ 0.40、失败题数 = 0：

```powershell
uv run python -m docguard.engineering.regression `
  --fixture data/eval/s11-regression-v1.json `
  --config configs/engineering/s11-engineering-v1.json
```

GitHub Actions 执行 Python 全量测试、Ruff、前端测试/类型检查/构建、fixture 回归、Compose 镜像构建和真实 Qdrant service parity。任何指标低于门槛、Test 锁被改动或容器合同漂移都会非零退出。

```powershell
$env:UV_CACHE_DIR = Join-Path (Get-Location) '.uv-cache'
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\verify_s11.ps1 `
  -RerankerModelRoot data/models/s6-bge-reranker-base
```

若本机没有 Docker，脚本会明确输出 `S11_DOCKER_RUNTIME_SKIPPED`；这只代表本地静态合同通过，容器 build/service/persistence 证据必须由 GitHub Actions 补齐。
