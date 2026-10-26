import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import { useApi } from "../api/context";
import type { EvidenceResponse } from "../api/types";
import { EmptyState, ErrorState, LoadingState } from "../components/AsyncState";
import { StatusBadge } from "../components/StatusBadge";
import { getCachedEvidence } from "../state/evidenceCache";

function metric(label: string, value: number | null) { return value === null ? null : label.includes("#") ? <div className="score-card"><strong>{label}</strong></div> : <div className="score-card"><small>{label}</small><strong>{value.toFixed(4)}</strong></div>; }
export function EvidencePage() {
  const { evidenceId } = useParams(); const api = useApi(); const [data, setData] = useState<EvidenceResponse | null>(null); const [error, setError] = useState(""); const cached = evidenceId ? getCachedEvidence(evidenceId) : null;
  useEffect(() => { if (!evidenceId) return; const c = new AbortController(); setError(""); api.getEvidence(evidenceId, { signal: c.signal }).then(setData).catch((e: unknown) => { if (!(e instanceof DOMException && e.name === "AbortError")) setError(e instanceof Error ? e.message : "未知错误"); }); return () => c.abort(); }, [api, evidenceId]);
  if (!evidenceId) return <section className="page"><header className="page-header"><div><p className="eyebrow">SOURCE INSPECTOR</p><h1>证据检查</h1></div></header><EmptyState title="尚未选择证据">从技术问答的引用进入，可同时查看本次查询排名。</EmptyState></section>;
  return <section className="page"><header className="page-header"><div><p className="eyebrow">SOURCE INSPECTOR</p><h1>证据检查</h1></div>{cached && <StatusBadge tone={cached.used ? "good" : "neutral"}>{cached.used ? "已用于回答" : "未用于回答"}</StatusBadge>}</header>{error ? <ErrorState message={error} /> : !data ? <LoadingState label="正在读取规范证据…" /> : <><div className="panel evidence-head"><p className="breadcrumbs">{data.section_path.join(" › ")}</p><h2>{data.topic}</h2><div className="metadata"><span>PostgreSQL {data.postgresql_version}</span><span className="mono">{data.chunk_id}</span><span>{data.chunking_version}</span></div><a className="source-link" href={data.canonical_url} target="_blank" rel="noreferrer">打开 PostgreSQL 官方原文</a></div>{cached ? <div className="score-grid">{metric(`Dense #${cached.evidence.provenance.dense_rank ?? "—"}`, cached.evidence.provenance.dense_rank)}{metric(`BM25 #${cached.evidence.provenance.bm25_rank ?? "—"}`, cached.evidence.provenance.bm25_rank)}{metric("RRF", cached.evidence.provenance.rrf_score)}{metric("Rerank", cached.evidence.provenance.rerank_score)}</div> : <p className="notice">当前页面没有查询排名上下文，仅展示规范原文。</p>}<article className="panel source-text"><div className="card-kicker">CANONICAL CHUNK</div><p>{data.text}</p><footer className="mono">SHA-256 {data.content_sha256}</footer></article></>}</section>;
}
