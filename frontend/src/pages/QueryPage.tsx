import { useEffect, useRef, useState, type FormEvent } from "react";
import { Link, useSearchParams } from "react-router-dom";
import type { PostgreSqlVersion } from "../api/types";
import type { KnowledgeBase } from "../api/discovery";
import { useApi } from "../api/context";
import { useQuerySession } from "../state/querySession";
import { ErrorState, LoadingState } from "../components/AsyncState";
import { StatusBadge } from "../components/StatusBadge";
import { CitedAnswer, citationLabel } from "../components/CitedAnswer";

const examples = ["VACUUM 有哪些主要选项？", "PostgreSQL 17 与 18 的 pg_dump 行为有什么差异？"];
export function QueryPage() {
  const s = useQuerySession();
  const { question, setQuestion, version, setVersion, result, error, loading, copied, submit, cancel, copy, knowledgeBaseId, setKnowledgeBaseId, knowledgeAnswer, answeredKnowledgeBaseId, generateKnowledgeAnswer, setGenerateKnowledgeAnswer } = s;
  const api = useApi(); const [params] = useSearchParams();
  const [bases, setBases] = useState<KnowledgeBase[]>([]);
  const [basesError, setBasesError] = useState("");
  const resultStart = useRef<HTMLDivElement>(null);
  useEffect(() => { const kb = params.get("kb"); if (kb) setKnowledgeBaseId(kb); }, [params, setKnowledgeBaseId]);
  useEffect(() => {
    const c = new AbortController();
    api.listKnowledgeBases({ signal: c.signal }).then(r => setBases(r.knowledge_bases)).catch((e: unknown) => { if (!c.signal.aborted) setBasesError(e instanceof Error ? e.message : "列表不可用"); });
    return () => c.abort();
  }, [api]);
  function onSubmit(event: FormEvent) { event.preventDefault(); void submit(); }
  const imported = Boolean(knowledgeBaseId);
  const answer = imported && answeredKnowledgeBaseId === knowledgeBaseId ? knowledgeAnswer : null;
  const displayedResponse = imported ? answer : result;
  useEffect(() => {
    if (!displayedResponse || loading) return;
    const reducedMotion = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
    resultStart.current?.scrollIntoView?.({ behavior: reducedMotion ? "auto" : "smooth", block: "start" });
  }, [displayedResponse, loading]);
  const pgCitations = result?.citations.map(c => ({ ...c, label: citationLabel(c.citation_label ?? "") ?? citationLabel(c.evidence_id) ?? (() => { const index = result.evidence.findIndex(e => e.chunk_id === c.chunk_id); return index >= 0 ? citationLabel(result.evidence[index].citation_label ?? "") ?? "E" + String(index + 1).padStart(2, "0") : undefined; })() })) ?? [];
  const pgLabels = new Set(pgCitations.flatMap(c => c.label ? [c.label] : []));
  const genericLabels = new Set(answer?.citations.map(c => citationLabel(c.evidence_id)).filter((v): v is string => Boolean(v)) ?? []);
  return <section className="page">
    <header className="page-header"><div><p className="eyebrow">TRACEABLE ANSWERS</p><h1>技术问答</h1><p className="lede">选择知识库，查询技术文档并核对引用。</p></div></header>
    <form className="panel query-form" onSubmit={onSubmit}>
      <label htmlFor="query-source">查询知识库</label>
      <select id="query-source" value={knowledgeBaseId} disabled={loading} onChange={e => setKnowledgeBaseId(e.target.value)}>
        <option value="">PostgreSQL 冻结知识库</option>
        {knowledgeBaseId && !bases.some(b => b.knowledge_base_id === knowledgeBaseId) && <option value={knowledgeBaseId}>{knowledgeBaseId}</option>}
        {bases.map(b => <option key={b.knowledge_base_id} value={b.knowledge_base_id}>{b.knowledge_base_id} · {b.product} {b.version ?? ""}</option>)}
      </select>
      {basesError && <p className="notice">导入知识库列表暂不可用：{basesError}</p>}
      {imported && <><label className="check-row"><input type="checkbox" checked={generateKnowledgeAnswer} disabled={loading} onChange={e => setGenerateKnowledgeAnswer(e.target.checked)} />使用大模型生成回答（调用已配置的 DeepSeek API）</label><p className="muted">{generateKnowledgeAnswer ? "先检索证据，再由模型组织回答并校验引用。" : "仅返回检索原文，不调用模型。"} 通用文档的拒答置信度尚未校准。</p></>}
      <label htmlFor="question">问题</label>
      <textarea id="question" value={question} maxLength={2000} rows={4} onChange={e => setQuestion(e.target.value)} placeholder={imported ? "输入所选知识库中的技术问题…" : "输入 PostgreSQL 技术问题…"} />
      <div className="form-row">
        {!imported && <><label htmlFor="version">目标版本</label><select id="version" value={version} onChange={e => setVersion(e.target.value as PostgreSqlVersion | "")}><option value="">由后端判断</option><option>17.11</option><option>18.6</option></select></>}
        <button type="submit" disabled={loading}>开始检索</button>{loading && <button type="button" onClick={cancel}>取消</button>}
      </div>
      {!imported && <div className="examples" aria-label="示例问题">{examples.map(x => <button type="button" key={x} onClick={() => setQuestion(x)}>{x}</button>)}</div>}
    </form>
    {loading && <LoadingState label={imported && !generateKnowledgeAnswer ? "正在检索文档原文…" : "正在检索、判断并生成…"} />}
    {error && <ErrorState message={error} />}
    {answer && <div className="result-stack query-result" ref={resultStart}>
      {answer.retrieval?.method === "full" && <p className="notice">当前知识库尚未完成领域拒答校准；检索追踪可点击引用进入证据检查查看。</p>}
      {answer.decision && <article className="panel decision-panel"><div><span className="card-kicker">BACKEND DECISION</span><h2>{answer.decision.action.toUpperCase()}</h2></div><StatusBadge tone={answer.decision.action === "answer" ? "good" : "warn"}>{answer.decision.reason_code ?? "ANSWER"}</StatusBadge><p className="mono">规则：{answer.decision.triggered_rule} · 置信度 {answer.decision.confidence === null ? "未校准" : answer.decision.confidence.toFixed(3)}</p></article>}
      <article className="panel answer-panel"><h2>{answer.action === "refuse" ? "无法基于当前文档回答" : answer.action === "clarify" ? "需要补充信息" : answer.mode === "generated" ? "回答" : "检索原文摘录"}</h2><p className="muted">知识库：{answeredKnowledgeBaseId}</p>{answer.action !== "answer" ? <p>{answer.explanation ?? `未找到匹配证据：${answer.reason ?? "evidence_insufficient"}`}</p> : <><CitedAnswer text={answer.answer ?? ""} labels={genericLabels} /><button type="button" onClick={copy}>{copied ? "已复制" : "复制回答"}</button></>}</article>
      {answer.model_info && <footer className="run-meta"><span>模型 {answer.model_info.model}</span><span>输入 {answer.model_info.input_tokens} · 输出 {answer.model_info.output_tokens} token</span><span>request {answer.model_info.request_id}</span></footer>}
      {answer.action !== "answer" && Boolean(answer.evidence?.length) && <details className="panel"><summary>查看未用于作答的检索候选</summary><p className="muted">这些片段仅为检索结果，未被声明为回答依据。</p>{answer.evidence?.map(e => <p key={e.chunk_id}><Link to={"/evidence?kb=" + encodeURIComponent(answeredKnowledgeBaseId) + "&chunk=" + encodeURIComponent(e.chunk_id)}>检索候选 {e.evidence_id} · {e.section_path.join(" / ") || e.chunk_id}</Link></p>)}</details>}
      <section><h2>引用证据 <span className="muted">{answer.citations.length}</span></h2><div className="evidence-list">{answer.citations.map(c => <article className="panel evidence-row knowledge-evidence" key={c.evidence_id} id={"citation-" + citationLabel(c.evidence_id)} tabIndex={-1}><Link to={"/evidence?kb=" + encodeURIComponent(answeredKnowledgeBaseId) + "&chunk=" + encodeURIComponent(c.chunk_id)}><strong>{c.evidence_id} · {c.section_path.join(" / ") || c.chunk_id}</strong></Link><small>{c.product} {c.version ?? ""} · {c.text.slice(0, 160)}</small><a className="citation-back" href="#answer-text">返回回答</a></article>)}</div></section>
    </div>}
    {!imported && result && <div className="result-stack query-result" ref={resultStart}>
      <article className="panel decision-panel"><div><span className="card-kicker">BACKEND DECISION</span><h2>{result.decision.action.toUpperCase()}</h2></div><StatusBadge tone={result.decision.action === "answer" ? "good" : result.decision.action === "error" ? "bad" : "warn"}>{result.decision.reason_code ?? "ANSWER"}</StatusBadge><p className="mono">规则：{result.decision.triggered_rule} · 置信度 {result.decision.confidence.toFixed(3)}</p></article>
      {result.answer !== null && <article className="panel answer-panel" data-testid="generated-answer"><h2>回答</h2><CitedAnswer text={result.answer} labels={pgLabels} /><button type="button" onClick={copy}>{copied ? "已复制" : "复制回答"}</button></article>}
      <section><h2>引用证据 <span className="muted">{result.citations.length}</span></h2>{pgCitations.length ? <div className="evidence-list">{pgCitations.map(c => <article className="panel evidence-row" key={c.evidence_id} id={c.label ? "citation-" + c.label : undefined} tabIndex={-1}><span className="mono">{c.label ?? "编号未提供"} · {c.postgresql_version}</span><Link to={"/evidence/" + encodeURIComponent(c.evidence_id)}><strong>{c.section_path.join(" / ")}</strong></Link><small>{c.text.slice(0, 160)}</small><a className="citation-back" href="#answer-text">返回回答</a></article>)}</div> : <p className="muted">本次响应没有引用。</p>}</section>
      <footer className="run-meta"><span>route {result.route}</span>{Object.entries(result.latency_ms).map(([k,v]) => <span key={k}>{k} {v.toFixed(1)} ms</span>)}{result.model_info && <span>{result.model_info.model}</span>}</footer>
    </div>}
  </section>;
}
