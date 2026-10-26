# frontend

S10 前端用于呈现知识库、问答、证据链与评测结果。界面只消费 S9 API 的真实状态和数据，不在浏览器中推断拒答结论或写死实验指标。

本地开发先在仓库根目录运行 `uv run docguard-api`，再进入本目录执行 `npm ci` 和 `npm run dev`。Vite 固定监听 `127.0.0.1:5173`，并将 `/api` 代理到 `127.0.0.1:8000`。缺少后端或本地制品时，页面会显示明确 error/degraded 状态；索引重建和在线评测执行器将在 S11 接入。

验证命令为 `npm test -- --run`、`npm run typecheck` 和 `npm run build`。仓库级门禁使用 `scripts/verify_s10.ps1`，不会调用真实生成 API，也不会解锁 Test。
