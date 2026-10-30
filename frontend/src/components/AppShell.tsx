import { useEffect, useState, type ReactNode } from "react";
import { NavLink, useLocation } from "react-router-dom";
import { useApi } from "../api/context";
import type { HealthResponse } from "../api/types";
import { StatusBadge } from "./StatusBadge";

const links = [["/knowledge", "知识库", "01"], ["/query", "技术问答", "02"], ["/evidence", "证据检查", "03"], ["/evaluations", "评测看板", "04"]] as const;

function ServiceHealth() {
  const api = useApi(); const [health, setHealth] = useState<HealthResponse | null>(null); const [failed, setFailed] = useState(false);
  useEffect(() => { const controller = new AbortController(); api.getHealth({ signal: controller.signal }).then(setHealth).catch((error: unknown) => { if (!(error instanceof DOMException && error.name === "AbortError")) setFailed(true); }); return () => controller.abort(); }, [api]);
  if (failed) return <StatusBadge tone="bad">API 不可达</StatusBadge>;
  if (!health) return <StatusBadge tone="neutral">连接中</StatusBadge>;
  return <StatusBadge tone={health.status === "ok" ? "good" : "warn"}>{health.status === "ok" ? "服务正常" : "服务降级"}</StatusBadge>;
}

export function AppShell({ children }: { children: ReactNode }) {
  const location = useLocation();
  return <div className="app-shell"><a className="skip-link" href="#main-content">跳到主要内容</a><aside className="sidebar" aria-label="主导航"><div className="brand"><span className="brand-mark">DG</span><div><strong>DocGuard-RAG</strong><small>CONTROL PLANE</small></div></div><nav>{links.map(([to, label, index]) => <NavLink key={to} to={to} className={({ isActive }) => isActive || (to === "/knowledge" && location.pathname === "/discovery") ? "nav-link active" : "nav-link"}><span>{index}</span>{label}</NavLink>)}</nav><div className="system-health"><small>系统状态</small><ServiceHealth /></div></aside><main id="main-content" tabIndex={-1}>{children}</main></div>;
}
