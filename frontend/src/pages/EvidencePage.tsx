import { useEffect, useState } from "react";
import { Link, useParams, useSearchParams } from "react-router-dom";
import { useApi } from "../api/context";
import type { EvidenceResponse } from "../api/types";
import { EmptyState, ErrorState, LoadingState } from "../components/AsyncState";
import { StatusBadge } from "../components/StatusBadge";
import { getCachedEvidence, getKnowledgeEvidence } from "../state/evidenceCache";
import { sourceUrl } from "../api/discovery";
import { RetrievalScores } from "../components/RetrievalScores";

export function EvidencePage() {
  const [params] = useSearchParams();
  const kb = params.get("kb"), chunk = params.get("chunk");
  const { evidenceId } = useParams(); const api = useApi(); const [data, setData] = useState<EvidenceResponse | null>(null); const [error, setError] = useState(""); const cached = evidenceId ? getCachedEvidence(evidenceId) : null;
  useEffect(() => { if (!evidenceId) return; const c = new AbortController(); setError(""); api.getEvidence(evidenceId, { signal: c.signal }).then(setData).catch((e: unknown) => { if (!(e instanceof DOMException && e.name === "AbortError")) setError(e instanceof Error ? e.message : "未知错误"); }); return () => c.abort(); }, [api, evidenceId]);
  if (kb && chunk) {
    const evidence = getKnowledgeEvidence(kb, chunk);
    const href = evidence && sourceUrl(evidence.url);
    return <section className="page"><header className="page-header"><div><p className="eyebrow">SOURCE INSPECTOR</p><h1>证据检查</h1></div></header>{evidence ? <><article className="panel evidence-head"><h2>{evidence.section_path.join(" / ") || evidence.chunk_id}</h2><p>{evidence.product} {evidence.version ?? "未指定版本"} · 知识库 {kb}</p><p className="mono">{evidence.chunk_id}</p>{href && <a className="source-link" href={href} target="_blank" rel="noopener noreferrer">打开文档原文</a>}</article><RetrievalScores provenance={evidence.provenance} /><article className="panel source-text"><p>{evidence.text}</p><p className="muted">本次检索的真实排名与模型原始分数，不代表答案正确概率；新领域拒答阈值单独校准。</p></article></> : <EmptyState title="当前证据上下文已失效"><Link to={"/query?kb=" + encodeURIComponent(kb)}>返回问答重新检索</Link></EmptyState>}</section>;
  }
  if (!evidenceId) return <section className="page"><header className="page-header"><div><p className="eyebrow">SOURCE INSPECTOR</p><h1>证据检查</h1></div></header><EmptyState title="尚未选择证据">从技术问答的引用进入，可同时查看本次查询排名。</EmptyState></section>;
  return <section className="page"><header className="page-header"><div><p className="eyebrow">SOURCE INSPECTOR</p><h1>证据检查</h1></div>{cached && <StatusBadge tone={cached.used ? "good" : "neutral"}>{cached.used ? "已用于回答" : "未用于回答"}</StatusBadge>}</header>{error ? <ErrorState message={error} /> : !data ? <LoadingState label="正在读取规范证据…" /> : <><div className="panel evidence-head"><p className="breadcrumbs">{data.section_path.join(" › ")}</p><h2>{data.topic}</h2><div className="metadata"><span>PostgreSQL {data.postgresql_version}</span><span className="mono">{data.chunk_id}</span><span>{data.chunking_version}</span></div><a className="source-link" href={data.canonical_url} target="_blank" rel="noreferrer">打开 PostgreSQL 官方原文</a></div>{cached ? <RetrievalScores provenance={cached.evidence.provenance} /> : <p className="notice">当前页面没有查询排名上下文，仅展示规范原文。</p>}<article className="panel source-text"><div className="card-kicker">CANONICAL CHUNK</div><p>{data.text}</p><footer className="mono">SHA-256 {data.content_sha256}</footer></article></>}</section>;
}
