import { createContext, useCallback, useContext, useEffect, useRef, useState, type ReactNode } from "react";
import { ApiError } from "../api/client";
import { useApi } from "../api/context";
import { canResubmitDiscovery, type DiscoveryCapabilities, type DiscoveryEvent, type DiscoveryTask } from "../api/discovery";
import { suggestKnowledgeBaseId } from "./knowledgeBaseName";

export const DISCOVERY_STORAGE = "docguard.discovery.v1";
const initialDraft = { documentType: "user_guide", importScope: "collection", goal: "", product: "", version: "", urls: "", officialOnly: true, reply: "", replyUrls: "", replyProduct: "", replyVersion: "", allowUnversioned: false, kbId: "" };
type Draft = typeof initialDraft;
function restore(): { taskId: string; draft: Draft } {
  try {
    const saved = JSON.parse(sessionStorage.getItem(DISCOVERY_STORAGE) ?? "{}");
    const draft = { ...initialDraft };
    for (const key of Object.keys(initialDraft) as (keyof Draft)[]) {
      if (typeof saved.draft?.[key] === typeof initialDraft[key]) Object.assign(draft, { [key]: saved.draft[key] });
    }
    return { taskId: typeof saved.taskId === "string" ? saved.taskId : "", draft };
  } catch { return { taskId: "", draft: initialDraft }; }
}
const message = (e: unknown) => e instanceof ApiError && e.status === 409 ? "候选修订或任务状态已变化，请重新核对后确认。" : e instanceof Error ? e.message : "请求失败，请稍后重试。";
const urls = (value: string) => [...new Set(value.split(/\r?\n/).map(v => v.trim()).filter(Boolean))];
const active = (task: DiscoveryTask | null) => task?.status === "discovering" || task?.status === "importing";

function useDiscoveryState() {
  const api = useApi();
  const [saved] = useState(restore);
  const [taskId, setTaskId] = useState(saved.taskId);
  const taskIdRef = useRef(taskId); taskIdRef.current = taskId;
  const [draft, setDraft] = useState(saved.draft);
  const [task, setTask] = useState<DiscoveryTask | null>(null);
  const taskRef = useRef<DiscoveryTask | null>(null);
  const [caps, setCaps] = useState<DiscoveryCapabilities | null>(null);
  const [capsError, setCapsError] = useState("");
  const [error, setError] = useState("");
  const [connection, setConnection] = useState("");
  const [busy, setBusy] = useState(false);
  const mutation = useRef(false);
  const [selected, setSelected] = useState<string[]>([]);
  const autoKbId = useRef("");
  useEffect(() => {
    if (task?.status !== "awaiting_selection" || !task.candidates.length) return;
    const suggestion = suggestKnowledgeBaseId(task, selected);
    const previous = autoKbId.current;
    autoKbId.current = suggestion;
    setDraft(value => !value.kbId || value.kbId === previous ? { ...value, kbId: suggestion } : value);
  }, [task, selected]);
  const [events, setEvents] = useState<DiscoveryEvent[]>([]);
  const cursor = useRef(0);
  const approval = useRef<{ signature: string; key: string } | null>(null);
  const [storageError, setStorageError] = useState("");
  const change = useCallback((patch: Partial<Draft>) => setDraft(v => ({ ...v, ...patch })), []);

  useEffect(() => {
    try { sessionStorage.setItem(DISCOVERY_STORAGE, JSON.stringify({ taskId, draft })); setStorageError(""); }
    catch { setStorageError("浏览器存储不可用，刷新后可使用任务编号手动恢复。"); }
  }, [taskId, draft]);
  useEffect(() => {
    const controller = new AbortController();
    api.getDiscoveryCapabilities({ signal: controller.signal }).then(setCaps).catch(e => { if (!controller.signal.aborted) setCapsError(message(e)); });
    return () => controller.abort();
  }, [api]);
  const accept = useCallback((value: DiscoveryTask) => {
    const previous = taskRef.current;
    if (previous && (previous.revision !== value.revision || previous.task_id !== value.task_id)) setSelected([]);
    taskRef.current = value; setTask(value);
    if (value.job?.knowledge_base_id && previous?.job?.job_id !== value.job.job_id) change({ kbId: value.job.knowledge_base_id });
  }, [change]);
  const refresh = useCallback(async () => {
    const id = taskIdRef.current; if (!id) return;
    try { const value = await api.getDiscoveryTask(id); if (taskIdRef.current === id) { accept(value); setError(""); } }
    catch (e) { if (taskIdRef.current === id) setError(message(e)); }
  }, [api, accept]);
  useEffect(() => {
    if (!taskId || taskRef.current?.task_id === taskId) return;
    const controller = new AbortController();
    api.getDiscoveryTask(taskId, { signal: controller.signal }).then(value => { if (!controller.signal.aborted) accept(value); }).catch(e => { if (!controller.signal.aborted) setError(message(e)); });
    return () => controller.abort();
  }, [taskId, api, accept]);

  const running = active(task);
  useEffect(() => {
    if (!taskId) return;
    let stopped = false, polling = false;
    const controller = new AbortController();
    const poll = async () => {
      if (polling || stopped) return; polling = true;
      try { const value = await api.getDiscoveryTask(taskId, { signal: controller.signal }); if (!stopped) accept(value); }
      catch (e) { if (!stopped) setConnection(message(e)); }
      finally { polling = false; }
    };
    const timer = running ? window.setInterval(() => { void poll(); }, 1500) : undefined;
    let reconnect: ReturnType<typeof setTimeout> | undefined;
    const stream = async () => {
      try {
        await api.watchDiscoveryEvents(taskId, cursor.current, event => {
          if (stopped || event.event_id <= cursor.current) return;
          cursor.current = event.event_id;
          setEvents(rows => [...rows, event].slice(-100)); setConnection(""); void poll();
        }, { signal: controller.signal });
      } catch (e) { if (!stopped) setConnection(message(e)); }
      if (!stopped && running) { void poll(); reconnect = setTimeout(() => { void stream(); }, 2500); }
    };
    void stream();
    return () => { stopped = true; controller.abort(); clearInterval(timer); clearTimeout(reconnect); };
  }, [api, taskId, running, accept]);

  function attach(value: DiscoveryTask) {
    taskIdRef.current = value.task_id; setTaskId(value.task_id); setSelected([]); cursor.current = 0; setEvents([]); accept(value);
  }
  async function mutate(action: () => Promise<void>) {
    if (mutation.current) return;
    mutation.current = true; setBusy(true); setError("");
    try { await action(); }
    catch (e) {
      setError(message(e));
      if (e instanceof ApiError && e.status === 409) {
        setSelected([]); approval.current = null;
        const id = taskIdRef.current;
        try { if (id) { const value = await api.getDiscoveryTask(id); if (taskIdRef.current === id) accept(value); } } catch { /* Keep the conflict visible until a successful manual refresh. */ }
      }
    }
    finally { mutation.current = false; setBusy(false); }
  }
  async function create() {
    const product = draft.product.trim();
    if (!product || product.toLowerCase() === "unspecified") { setError("请输入产品名称"); return; }
    const target = product + (draft.version.trim() ? " " + draft.version.trim() : "");
    const goal = draft.goal.trim() || `查找 ${target} 的官方文档`;
    if (caps && !caps.agent_enabled && !urls(draft.urls).length) { setError("当前未配置自动搜索，请填写文档网址。"); return; }
    await mutate(async () => attach(await api.createDiscoveryTask({ goal, product, version: draft.version.trim() || null, urls: urls(draft.urls), official_only: draft.officialOnly, document_type: draft.documentType })));
  }
  async function resubmit() {
    if (!task || !canResubmitDiscovery(task)) return;
    const request = { goal: task.goal, product: task.product, version: task.version, urls: task.urls ?? [], official_only: task.official_only ?? true, document_type: task.document_type ?? "user_guide" };
    await mutate(async () => {
      const next = await api.createDiscoveryTask(request);
      change({ goal: request.goal, product: request.product, version: request.version ?? "", urls: request.urls.join("\n"), officialOnly: request.official_only, documentType: request.document_type, reply: "", replyUrls: "", replyProduct: "", replyVersion: "", allowUnversioned: false });
      approval.current = null;
      attach(next);
    });
  }
  async function reply() {
    if (task?.status === "failed" && task.stop_reason !== "agent_model_failed" && !draft.replyUrls.trim()) { setError("请填写需要重新核查的文档网址。"); return; }
    if (!task || !(draft.reply.trim() || draft.replyUrls.trim() || draft.replyProduct.trim() || draft.replyVersion.trim() || draft.allowUnversioned)) { setError("请填写补充说明、产品版本或文档网址"); return; }
    await mutate(async () => { accept(await api.replyDiscoveryTask(task.task_id, { message: draft.reply.trim() || undefined, urls: urls(draft.replyUrls), product: draft.replyProduct.trim() || undefined, version: draft.replyVersion.trim() || undefined, allow_unversioned: draft.allowUnversioned })); change({ reply: "", replyUrls: "", replyProduct: "", replyVersion: "", allowUnversioned: false }); });
  }
  async function approve() {
    if (!task || !selected.length) return;
    if (!/^[a-z0-9][a-z0-9_-]{0,63}$/.test(draft.kbId)) { setError("知识库标识需为 1–64 位小写字母、数字、下划线或短横线"); return; }
    if (draft.importScope === "collection" && selected.length > 1 && task.candidates.some(c => selected.includes(c.candidate_id) && c.collection)) { setError("完整集合导入请只选择一个文档入口；无需选择其下各章节。"); return; }
    await mutate(async () => {
      const collection = selected.length === 1 && task.candidates.find(c => c.candidate_id === selected[0])?.collection;
      const importScope = draft.importScope === "collection" && collection ? "collection" : "pages";
      const signature = JSON.stringify([task.task_id, task.revision, [...selected].sort(), draft.kbId, importScope]);
      if (approval.current?.signature !== signature) approval.current = { signature, key: `approve-${crypto.randomUUID()}` };
      const job = await api.approveDiscoveryTask(task.task_id, { candidate_ids: selected, revision: task.revision, knowledge_base_id: draft.kbId, idempotency_key: approval.current.key, import_scope: importScope });
      accept({ ...task, status: "importing", job });
    });
  }
  async function retryImport() {
    if (!task?.job || task.job.status !== "failed") return;
    await mutate(async () => { const job = await api.retryImport(task.job!.job_id); accept({ ...task, status: "importing", job }); });
  }
  async function cancel() { if (task) await mutate(async () => accept(await api.cancelDiscoveryTask(task.task_id))); }
  async function restoreTask(id: string) {
    if (!id.trim()) return;
    await mutate(async () => attach(await api.getDiscoveryTask(id.trim())));
  }
  return { task, taskId, draft, change, caps, capsError, error, connection, busy, selected, setSelected, events, create, resubmit, reply, approve, retryImport, cancel, refresh, restoreTask, storageError };
}

const DiscoveryContext = createContext<ReturnType<typeof useDiscoveryState> | null>(null);
export function DiscoverySessionProvider({ children }: { children: ReactNode }) { return <DiscoveryContext.Provider value={useDiscoveryState()}>{children}</DiscoveryContext.Provider>; }
export function useDiscoverySession() {
  const value = useContext(DiscoveryContext); if (!value) throw new Error("DiscoverySessionProvider is required"); return value;
}
