import { useCallback, useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { useApi } from "../api/context";
import type { CorpusResponse, DependencyHealth } from "../api/types";
import type { KnowledgeBase } from "../api/discovery";
import { EmptyState, ErrorState, LoadingState } from "../components/AsyncState";
import { StatusBadge } from "../components/StatusBadge";

export function KnowledgePage() {
  const api = useApi();
  const [corpora, setCorpora] = useState<CorpusResponse[] | null>(null);
  const [deps, setDeps] = useState<DependencyHealth[]>([]);
  const [error, setError] = useState(""); const [rebuild, setRebuild] = useState("");
  const [bases, setBases] = useState<KnowledgeBase[] | null>(null); const [basesError, setBasesError] = useState("");
  const [deleteId, setDeleteId] = useState("");
  const [deleting, setDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState("");
  const [deleteNotice, setDeleteNotice] = useState("");
  async function removeBase(id: string) {
    if (deleting) return;
    setDeleting(true); setDeleteError(""); setDeleteNotice("");
    try {
      await api.deleteKnowledgeBase(id);
      setBases(rows => rows?.filter(row => row.knowledge_base_id !== id) ?? null);
      setDeleteId(""); setDeleteNotice(`已删除 ${id}`);
      window.dispatchEvent(new CustomEvent("docguard:knowledge-deleted", { detail: id }));
    } catch (e) { setDeleteError(e instanceof Error ? e.message : "删除失败"); }
    finally { setDeleting(false); }
  }
  const load = useCallback(() => {
    const c = new AbortController(); setError("");
    Promise.all([api.getCorpora({ signal: c.signal }), api.getHealth({ signal: c.signal })]).then(([a, h]) => { setCorpora(a.corpora); setDeps(h.dependencies); }).catch((e: unknown) => { if (!c.signal.aborted) setError(e instanceof Error ? e.message : "未知错误"); });
    return () => c.abort();
  }, [api]);
  const loadBases = useCallback(() => {
    const c = new AbortController(); setBasesError("");
    api.listKnowledgeBases({ signal: c.signal }).then(r => setBases(r.knowledge_bases)).catch((e: unknown) => { if (!c.signal.aborted) setBasesError(e instanceof Error ? e.message : "列表不可用"); });
    return () => c.abort();
  }, [api]);
  useEffect(load, [load]); useEffect(loadBases, [loadBases]);
  async function runRebuild() { setRebuild("正在提交重建任务…"); try { const result = await api.rebuildCorpora(); setRebuild("任务 " + result.job_id + " 已进入队列"); } catch (e) { setRebuild("重建失败：" + (e instanceof Error ? e.message : "未知错误")); } }
  return <section className="page">
    <header className="page-header"><div><p className="eyebrow">CORPUS CONTROL</p><h1>知识库管理</h1><p className="lede">管理文档来源、已导入知识库和索引状态。</p></div><div className="form-row"><Link className="action-link" to="/discovery">添加技术文档</Link><button type="button" onClick={runRebuild}>重建索引</button></div></header>
    <section><div className="version-row"><h2>已导入技术文档</h2><button type="button" onClick={() => { loadBases(); }}>刷新文档列表</button></div>
      {deleteNotice && <p className="notice" role="status">{deleteNotice}</p>}
      {deleteError && <p className="notice" role="alert">{deleteError}</p>}
      {basesError ? <ErrorState message={basesError} retry={loadBases} /> : bases === null ? <LoadingState label="正在读取导入文档…" /> : !bases.length ? <p className="muted">还没有导入技术文档，点击“添加技术文档”开始。</p> : <div className="corpus-grid">{bases.map(b => <article className="panel corpus-card" key={b.knowledge_base_id}>
        <div className="card-kicker">{b.product} {b.version ?? "未限定版本"}</div><h3>{b.knowledge_base_id}</h3>
        <div className="stat-pair"><div><strong>{b.document_count.toLocaleString()}</strong><span>文档</span></div><div><strong>{b.chunk_count.toLocaleString()}</strong><span>分块</span></div></div>
        <dl><dt>快照标识</dt><dd className="mono">{b.snapshot_id}</dd></dl><Link className="source-link" to={"/query?kb=" + encodeURIComponent(b.knowledge_base_id)}>向此知识库提问</Link>
        {deleteId === b.knowledge_base_id ? <div className="notice"><p>删除后将从列表和可查询知识库中移除，历史快照与任务记录保留。</p><div className="form-row"><button type="button" disabled={deleting} onClick={() => { void removeBase(b.knowledge_base_id); }}>{deleting ? "正在删除…" : "确认删除"}</button><button type="button" disabled={deleting} onClick={() => { setDeleteId(""); setDeleteError(""); }}>取消删除</button></div></div> : <button type="button" aria-label={`删除 ${b.knowledge_base_id}`} disabled={deleting} onClick={() => { setDeleteId(b.knowledge_base_id); setDeleteError(""); }}>删除文档</button>}
      </article>)}</div>}
    </section>
    <h2>PostgreSQL 冻结语料</h2>
    {rebuild && <p className="notice" role="status" aria-live="polite">{rebuild}</p>}
    {error ? <ErrorState message={error} retry={load} /> : corpora === null ? <LoadingState label="正在读取语料状态…" /> : corpora.length === 0 ? <EmptyState title="还没有可用语料">请先恢复冻结文档快照并构建索引。</EmptyState> : <div className="corpus-grid">{corpora.map(c => <article className="panel corpus-card" key={c.version}><div className="card-kicker">POSTGRESQL</div><div className="version-row"><h2>{c.version}</h2><StatusBadge tone={c.index_status === "ready" ? "good" : "bad"}>{c.index_status}</StatusBadge></div><div className="stat-pair"><div><strong>{c.document_count.toLocaleString()}</strong><span>文档</span></div><div><strong>{c.chunk_count.toLocaleString()}</strong><span>分块</span></div></div><dl><dt>快照 SHA-256</dt><dd className="mono">{c.snapshot_sha256 ?? "未提供"}</dd><dt>构建时间</dt><dd>{c.built_at ? new Date(c.built_at).toLocaleString("zh-CN") : "未提供"}</dd></dl></article>)}</div>}
    <section className="panel dependency-panel"><h2>依赖状态</h2>{deps.length ? <ul className="dependency-list">{deps.map(d => <li key={d.name}><span>{d.name}</span><StatusBadge tone={d.status === "available" ? "good" : "bad"}>{d.status}</StatusBadge>{d.detail && <small>{d.detail}</small>}</li>)}</ul> : <p className="muted">API 未返回依赖条目。</p>}</section>
  </section>;
}
