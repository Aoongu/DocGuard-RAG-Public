import { ImportProgress } from "../components/ImportProgress";
import { useLayoutEffect, useRef, useState, type CSSProperties } from "react";
import { Link } from "react-router-dom";
import { canResubmitDiscovery, sourceUrl, type DiscoveryCandidate } from "../api/discovery";
import { useDiscoverySession } from "../state/discoverySession";
import { StatusBadge } from "../components/StatusBadge";

const statuses: Record<string, string> = { discovering: "正在查找与核查", awaiting_input: "等待补充信息", awaiting_selection: "等待确认候选", importing: "正在导入", ready: "导入完成", failed: "任务失败", exhausted: "预算耗尽", cancelled: "已取消" };
const sourceLabels: Record<string, string> = { official_domain: "官方域名已核对", agent_assessed_official: "Agent 判断为官方 · 待你确认", unverified: "来源未确认", user_supplied_unverified: "用户提供 · 来源未确认" };
const versionLabels: Record<string, string> = { url_version_match: "网址或标题版本匹配", not_requested: "未要求指定版本", user_supplied_unverified: "用户指定 · 未核实", unknown: "版本未知", version_mismatch: "版本不符" };
const failureMessages: Record<string, string> = { dns_non_public: "文档域名解析到了非公网地址，常见原因是代理 Fake-IP。请启用应用的真实 DNS 解析，或调整 DNS 后补充网址重新核查。", dns_resolution_failed: "无法解析文档域名。请检查网络和 DNS 配置后补充网址重新核查。", https_download_failed: "网页连接失败，已进行有限重试和备用公网 IP 尝试。网络恢复后可重新提交。", robots_unavailable: "无法确认网站的 robots 访问政策，已停止导入。", robots_disallowed: "网站的 robots 政策不允许抓取此页。", not_html: "该网址没有返回可解析的 HTML 文档。" };
function Source({ value, children }: { value: string; children: React.ReactNode }) {
  const href = sourceUrl(value);
  return href ? <a className="source-link" href={href} target="_blank" rel="noopener noreferrer">{children}</a> : <span>{children}（链接不可用）</span>;
}
function CandidateInfo({ candidate }: { candidate: DiscoveryCandidate }) {
  return <><strong>{candidate.title}</strong><Source value={candidate.url}>{candidate.url}</Source><div className="metadata"><span>{sourceLabels[candidate.source_status] ?? candidate.source_status}</span><span>{candidate.version ?? "未指定版本"} · {versionLabels[candidate.version_status] ?? candidate.version_status}</span></div>
    {candidate.source_assessment && <div className="notice"><p>{candidate.source_assessment.reason}</p><p>页面链接关系不能独立证明域名归属。请核对来源与版本后再确认导入。</p><details><summary>查看 Agent 的来源判断依据</summary>{candidate.source_assessment.authority_pages.map(page => <div key={page.url}><Source value={page.url}>{page.url}</Source><small className="mono">SHA-256: {page.content_sha256}</small></div>)}</details></div>}
    <details><summary>正文快照 SHA-256</summary><code className="mono">{candidate.content_sha256}</code></details></>;
}

export function DiscoveryPage() {
  const s = useDiscoverySession(); const { draft, task } = s;
  const [restoreId, setRestoreId] = useState("");
  const leftPanel = useRef<HTMLElement>(null);
  const [expandedPanelHeight, setExpandedPanelHeight] = useState<number>();
  useLayoutEffect(() => {
    const panel = leftPanel.current;
    if (!panel) return;
    const measure = () => {
      const width = panel.getBoundingClientRect().width;
      if (!width) return;
      // Measure the fully expanded form without changing the user's open state.
      const expanded = panel.cloneNode(true) as HTMLElement;
      expanded.setAttribute("aria-hidden", "true");
      expanded.setAttribute("inert", "");
      expanded.querySelectorAll("[id]").forEach(node => node.removeAttribute("id"));
      expanded.querySelectorAll("details").forEach(node => { node.open = true; });
      Object.assign(expanded.style, { position: "fixed", visibility: "hidden", pointerEvents: "none", width: `${width}px`, height: "auto", maxHeight: "none", margin: "0", left: "0", top: "0" });
      panel.parentElement?.appendChild(expanded);
      const height = Math.ceil(expanded.getBoundingClientRect().height);
      expanded.remove();
      if (height) setExpandedPanelHeight(previous => previous === height ? previous : height);
    };
    measure();
    if (typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(measure);
    observer.observe(panel);
    return () => observer.disconnect();
  }, [task?.task_id, task?.status]);
  const canStart = !s.busy && (!task || ["ready", "failed", "exhausted", "cancelled"].includes(task.status));
  const canCancel = task && !["ready", "failed", "exhausted", "cancelled"].includes(task.status) && task.job?.status !== "publishing";
  const needsUrl = s.caps !== null && !s.caps.agent_enabled;
  const canEditSources = task && !task.job && ["awaiting_input", "awaiting_selection", "failed"].includes(task.status);
  const initialForm = <form className="query-form" onSubmit={e => { e.preventDefault(); void s.create(); }}>
        <div className="discovery-fields"><label>产品（必填）<input required maxLength={100} value={draft.product} onChange={e => s.change({ product: e.target.value })} placeholder="例如：Python、pandas、Java" /></label><label>版本（可选）<input maxLength={100} value={draft.version} onChange={e => s.change({ version: e.target.value })} placeholder="例如：3" /></label></div>
        <label>文档类型<select value={draft.documentType} onChange={e => s.change({ documentType: e.target.value })}><option value="user_guide">详细使用文档（默认）</option><option value="api_reference">API 参考</option><option value="getting_started">快速入门</option><option value="topic">具体主题</option></select></label>
        <label htmlFor="discovery-goal">文档需求（可选）</label><textarea id="discovery-goal" rows={3} maxLength={2000} value={draft.goal} onChange={e => s.change({ goal: e.target.value })} placeholder="留空默认查找该产品的官方文档；可填写具体主题，例如配置文件解析" /><small className="muted">留空时默认：查找 {draft.product.trim() || "产品名称"}{draft.version.trim() ? " " + draft.version.trim() : ""} 的官方文档。</small>
        <label className="check-row"><input type="checkbox" checked={draft.officialOnly} onChange={e => s.change({ officialOnly: e.target.checked })} />仅搜索官方来源</label>
        <label htmlFor="discovery-urls">{needsUrl ? "文档网址（必填，每行一个）" : "文档网址（可选，每行一个）"}</label><textarea id="discovery-urls" rows={2} value={draft.urls} onChange={e => s.change({ urls: e.target.value })} placeholder="https://docs.python.org/3/library/configparser.html" />
        <small className="muted">提供的网址作为明确来源处理，来源与版本会显示核查状态；只支持可读取的静态 HTML 正文。</small>
        <button disabled={!canStart}>{s.busy ? "正在提交…" : "查找文档"}</button>
      </form>;
  return <section className="page discovery-page">
    <header className="page-header"><div><p className="eyebrow">DOCUMENT DISCOVERY</p><h1>添加技术文档</h1><p className="lede">描述需求，核对文档，再导入专属知识库。</p></div><StatusBadge tone={s.caps?.agent_enabled ? "good" : "warn"}>{s.caps?.agent_enabled ? "自动查找已配置" : "网址导入模式"}</StatusBadge></header>
    {s.caps && !s.caps.agent_enabled && <p className="notice">自动查找未启用：{!s.caps.decision_model_configured ? "模型服务未配置；" : ""}{!s.caps.search_configured ? "搜索服务未配置；" : ""}请填写文档网址后导入。</p>}
    {s.capsError && <p className="notice">无法读取搜索服务状态：{s.capsError}</p>}
    {s.storageError && <p role="alert" className="notice">{s.storageError}</p>}
    {s.error && <p role="alert" className="notice">{s.error}</p>}
    {task?.job && <ImportProgress job={task.job} />}
    <div className="discovery-grid" style={{ "--discovery-panel-height": expandedPanelHeight ? `${expandedPanelHeight}px` : undefined } as CSSProperties}>
      <section ref={leftPanel} className="panel" aria-label="需求与补充">
        <h2>{task ? "1. 补充与调整" : "1. 描述需求"}</h2>
        {!task ? initialForm : <>
          <p className="muted">任务需求：{task.goal} · 产品：{task.product} · 版本：{task.version ?? "未指定"}</p>
          {task.pending_question && <p role="status" className="notice">{task.pending_question}</p>}
          {canEditSources && <form className="query-form" onSubmit={e => { e.preventDefault(); void s.reply(); }}><label htmlFor="discovery-reply">补充说明</label><textarea id="discovery-reply" rows={2} maxLength={2000} value={draft.reply} onChange={e => s.change({ reply: e.target.value })} /><div className="discovery-fields"><label>补充产品（可选）<input maxLength={100} value={draft.replyProduct} onChange={e => s.change({ replyProduct: e.target.value })} /></label><label>补充版本（可选）<input maxLength={100} value={draft.replyVersion} onChange={e => s.change({ replyVersion: e.target.value })} /></label></div><label className="check-row"><input type="checkbox" checked={draft.allowUnversioned} onChange={e => s.change({ allowUnversioned: e.target.checked })} />允许版本未知的文档</label><label htmlFor="reply-urls">补充网址（每行一个）</label><textarea id="reply-urls" rows={2} value={draft.replyUrls} onChange={e => s.change({ replyUrls: e.target.value })} /><button disabled={s.busy}>{task.status === "failed" ? task.stop_reason === "agent_model_failed" ? "继续当前任务" : "更新网址并重新核查" : "提交补充"}</button></form>}
          {!canEditSources && <p className="muted">{task.status === "ready" ? "本次导入已完成，可展开下方表单开始新任务。" : task.status === "discovering" || task.status === "importing" ? "任务正在处理，完成后可在此补充信息。" : "可展开下方表单开始新任务。"}</p>}
          <details className="restore-task"><summary>初始需求与新任务</summary>{initialForm}</details>
        </>}
<details className="restore-task"><summary>用任务编号恢复</summary><form className="query-form" onSubmit={e => { e.preventDefault(); void s.restoreTask(restoreId); }}><label htmlFor="restore-task-id">任务编号</label><input id="restore-task-id" value={restoreId} onChange={e => setRestoreId(e.target.value)} /><button disabled={s.busy}>恢复任务</button></form></details></section>
      <section className="panel discovery-results" aria-label="核查与确认"><div className="version-row"><h2>2. 核查与确认</h2>{task && <StatusBadge tone={task.status === "failed" ? "bad" : task.status === "ready" ? "good" : "neutral"}>{statuses[task.status] ?? task.status}</StatusBadge>}</div>
        {!task ? <p className="muted">{s.taskId ? "正在恢复已保存的任务…" : "提交需求后，在这里核对候选文档。"}</p> : <div className="result-stack">
          <div className="run-meta"><span className="mono">{task.task_id}</span><span>修订 {task.revision}</span></div>

          <div className="form-row"><button type="button" disabled={s.busy} onClick={() => { void s.refresh(); }}>刷新状态</button>{canResubmitDiscovery(task) && <button type="button" disabled={s.busy} onClick={() => { void s.resubmit(); }}>重新提交搜索</button>}{canCancel && <button type="button" disabled={s.busy} onClick={() => { void s.cancel(); }}>取消任务</button>}</div>
          {canResubmitDiscovery(task) && <p className="muted">刷新状态只读取进度；重新提交搜索会按原需求创建新任务，重新计算搜索预算，并保留旧任务日志。无需先取消或补充网址。</p>}
          {s.connection && <p role="status" className="muted">事件连接重试中，自动轮询状态。{s.connection}</p>}

          {task.status === "failed" && !task.job && <p className="notice">{task.stop_reason === "agent_model_failed" ? "模型决策失败。可填写补充说明后继续当前任务，历史记录与预算保留。" : "刷新状态只读取进度。可在左侧补充或替换网址，继续当前任务，无需取消重建。"}</p>}
          {task.stop_reason && <p className="muted">停止原因：{task.stop_reason}</p>}

          {task.status === "awaiting_selection" && <><p className="muted">选择文档入口即可，无需逐个挑选章节。完整集合会沿该目录中的链接自动收集静态 HTML 页面；最多 1000 页，超限或失败不会发布不完整知识库。</p><div className="candidate-list">{task.candidates.map(c => <article className="candidate-card" key={c.candidate_id}><label className="check-row"><input aria-label={`选择 ${c.title}`} type="checkbox" checked={s.selected.includes(c.candidate_id)} onChange={e => s.setSelected(ids => e.target.checked ? [...ids, c.candidate_id] : ids.filter(id => id !== c.candidate_id))} />{c.collection ? "选择此文档集合" : "选择此页"}</label><CandidateInfo candidate={c} />{c.collection && <p className="muted">集合范围：{c.collection.path_prefix}（包含其子目录）</p>}</article>)}</div><label className="query-form">导入范围<select value={draft.importScope} onChange={e => s.change({ importScope: e.target.value })}><option value="collection">完整文档集合（仅目录入口支持，请选择一个集合）</option><option value="pages">仅选中页面</option></select></label><label className="query-form">知识库标识<input value={draft.kbId} maxLength={64} onChange={e => s.change({ kbId: e.target.value })} placeholder="根据页面标题生成，可修改" /></label><button type="button" disabled={s.busy || !s.selected.length} onClick={() => { void s.approve(); }}>确认导入</button></>}
          {task.approved_manifest && <section><h3>已批准清单</h3><p>知识库：{task.approved_manifest.knowledge_base_id}</p>{task.approved_manifest.candidates.map(c => <article className="candidate-card" key={c.candidate_id}><CandidateInfo candidate={c} />{c.collection && <p className="muted">集合范围：{c.collection.path_prefix}（包含其子目录）</p>}</article>)}</section>}
          {task.job && <section aria-label="导入任务详情"><h3>导入任务详情</h3><p>{task.job.status === "succeeded" ? "已发布知识库快照" : task.job.status === "publishing" ? "正在发布快照，此时不能取消" : task.job.status === "failed" ? "导入失败" : "正在处理批准的页面…"}</p><p className="mono">{task.job.job_id}</p>{task.job.result && <p>{task.job.result.document_count} 页 · {task.job.result.chunk_count} 个分块</p>}{task.job.status === "failed" && <button type="button" disabled={s.busy} onClick={() => { void s.retryImport(); }}>重试导入（复用已下载页面）</button>}{task.job.failure_type && <p role="alert">{task.job.failure_type} · {task.job.failure_reason ?? "请检查后台日志"}。本次失败未发布新的知识库快照。{task.job.failure_url && <> 失败页面：{task.job.failure_url}</>}</p>}</section>}
          {!!task.failures.length && <section><h3>逐页失败</h3><ul>{task.failures.map((f, i) => <li key={`${f.url}-${i}`}><Source value={f.url}>{f.url}</Source>{failureMessages[f.reason_code] && <p>{failureMessages[f.reason_code]}</p>}<p>{f.error_type}</p><small>{f.reason_code}</small>{!!f.network_attempts?.length && <p className="muted">连接尝试：{f.network_attempts.length} 次；最后异常：{f.network_attempts.at(-1)?.phase} / {f.network_attempts.at(-1)?.error_type}</p>}</li>)}</ul></section>}
          {!!s.events.length && <details><summary>任务轨迹（最近 {s.events.length} 条）</summary><ol className="event-list" aria-label="任务轨迹列表" tabIndex={0}>{s.events.map(event => <li key={event.event_id}><strong>#{event.event_id} {event.kind}</strong><pre>{JSON.stringify(event.data, null, 2)}</pre></li>)}</ol></details>}
        </div>}
      </section>
    </div>
    <section className="panel"><p>导入的文档统一在知识库中管理，在技术问答中选择并查询。</p><Link className="action-link" to="/knowledge">返回列表</Link>{task?.status === "ready" && task.job?.knowledge_base_id && <p><Link className="action-link" to={"/query?kb=" + encodeURIComponent(task.job.knowledge_base_id)}>开始提问</Link></p>}</section>
  </section>;
}
