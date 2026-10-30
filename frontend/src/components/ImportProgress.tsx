import type { ImportJob } from "../api/discovery";

const stages = ["收集页面", "解析与切块", "关键词索引", "生成向量", "发布知识库"];
const phaseIndex: Record<string, number> = { collecting: 0, parsing: 1, chunking: 1, bm25: 2, indexing: 2, embedding: 3, dense_index: 3, publishing: 4, completed: 4 };
const labels: Record<string, string> = { queued: "准备导入", collecting: "正在收集页面", parsing: "正在解析页面", chunking: "正在解析与切块", bm25: "正在构建关键词索引", indexing: "正在构建检索索引", embedding: "正在生成向量", dense_index: "正在写入向量索引", publishing: "正在发布知识库", completed: "导入完成，可以开始提问" };
const descriptions: Record<string, string> = { queued: "任务已排队，准备读取文档。", collecting: "正在沿文档目录发现和读取章节，最终页数仍可能增加。", parsing: "正在提取网页正文、标题、表格和代码。", chunking: "正在按章节结构切块，并检查每块的长度上限。", bm25: "正在建立关键词检索索引。", embedding: "正在本地按批次编码；首次运行需要加载模型，完成一批后计数更新。", dense_index: "向量编码已完成，正在保存检索索引。", publishing: "正在校验并发布知识库快照，完成后即可提问。" };
const format = (value: number) => value.toLocaleString("zh-CN");

export function ImportProgress({ job }: { job: ImportJob }) {
  const p = job.progress;
  const done = job.status === "succeeded", failed = job.status === "failed", cancelled = job.status === "cancelled";
  const phase = done ? "completed" : job.status === "queued" ? "queued" : job.status === "publishing" ? "publishing" : p?.phase ?? "queued";
  const step = phaseIndex[phase] ?? 0;
  const total = p?.total ?? 0, completed = Math.max(0, Math.min(p?.completed ?? 0, total));
  const counted = total > 0 && (p?.unit === "pages" || p?.unit === "chunks");
  const percent = Math.floor(100 * completed / Math.max(total, 1));
  const title = done ? labels.completed : failed ? `导入失败 · ${stages[step]}` : cancelled ? "导入已取消" : labels[phase] ?? "正在导入";
  const details = done ? `${format(job.result?.document_count ?? 0)} 页文档 · ${format(job.result?.chunk_count ?? 0)} 个分块` : phase === "collecting" ? `已读取 ${format(p?.completed_pages ?? 0)} 页 · 已发现 ${format(p?.discovered_pages ?? 0)} 页` : counted ? `${format(completed)} / ${format(total)} ${p?.unit === "chunks" ? "个分块" : "页文档"}` : p?.chunk_count !== undefined ? `${format(p.chunk_count)} 个分块` : "等待当前步骤完成";
  const source = p?.current_url ?? p?.last_url;
  return <section className={`import-progress ${done ? "is-complete" : failed ? "is-failed" : ""}`} aria-label="导入进度">
    <div className="import-progress-heading"><div><span className="card-kicker">知识库导入</span><h2>{title}</h2></div><span className="import-progress-badge">{done ? "全部完成" : failed ? "需要处理" : cancelled ? "已停止" : `阶段 ${step + 1} / 5`}</span></div>
    <ol className="import-steps" aria-label="导入阶段">{stages.map((name, index) => <li key={name} className={done || index < step ? "is-done" : index === step ? "is-current" : ""} aria-current={!done && index === step ? "step" : undefined}><span>{done || index < step ? "✓" : index + 1}</span>{name}</li>)}</ol>
    <div className="import-progress-count" aria-live="polite">{details}<small>{done ? "已发布" : counted && !failed && !cancelled ? `当前阶段 ${percent}%` : ""}</small></div>
    <progress aria-label="当前阶段进度" max={done ? 1 : counted ? total : 1} value={done ? 1 : counted ? completed : failed || cancelled ? 0 : undefined} />
    <p>{failed ? "任务在此阶段失败，新知识库快照尚未发布。可查看下方错误后重试。" : cancelled ? "任务已停止。" : descriptions[phase]}</p>
    {source && !done && <small className="import-progress-source">当前页面：{source}</small>}
    {p?.updated_at && !done && <small className="muted">最近更新：{new Date(p.updated_at).toLocaleTimeString("zh-CN", { hour12: false })}</small>}
  </section>;
}
