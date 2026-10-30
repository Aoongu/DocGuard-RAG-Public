import { act, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { expect, test, vi } from "vitest";
import { AppRoutes } from "../App";
import { ApiProvider } from "../api/context";
import { ApiError } from "../api/client";
import type { DiscoveryTask, KnowledgeAnswer } from "../api/discovery";
import { fakeApi } from "../test/fakeApi";

export const candidate = { candidate_id: "c1", url: "https://docs.python.org/3/library/configparser.html", title: "ConfigParser", content_sha256: "abc", version: "3", version_status: "url_version", source_status: "official_domain", status: "inspected" };
export const task: DiscoveryTask = { task_id: "discover-test", goal: "Python 配置文档", product: "Python", version: "3", revision: 2, status: "awaiting_selection", candidates: [candidate], failures: [], pending_question: null, job: null };
const caps = { html_import: true, decision_model_configured: true, search_configured: true, agent_enabled: true };

test("reuses the left panel for supplements after submission and keeps results on the right", async () => {
  mount({ createDiscoveryTask: async () => task });
  await userEvent.type(screen.getByLabelText("产品（必填）"), "pandas");
  await userEvent.click(screen.getByRole("button", { name: "查找文档" }));
  const left = await screen.findByRole("region", { name: "需求与补充" });
  const right = screen.getByRole("region", { name: "核查与确认" });
  expect(within(left).getByLabelText("补充说明")).toBeVisible();
  expect(within(right).queryByLabelText("补充说明")).not.toBeInTheDocument();
  expect(within(right).getByText("ConfigParser")).toBeVisible();
  expect(screen.getByLabelText("产品（必填）")).not.toBeVisible();
  await userEvent.click(within(left).getByText("初始需求与新任务"));
  expect(screen.getByLabelText("产品（必填）")).toBeVisible();
});

test("suggests a title based knowledge base ID and preserves a manual edit", async () => {
  mount({ createDiscoveryTask: async () => task });
  await userEvent.type(screen.getByLabelText("产品（必填）"), "Python");
  await userEvent.click(screen.getByRole("button", { name: "查找文档" }));
  const id = await screen.findByLabelText("知识库标识");
  expect(id).toHaveValue("python-configparser");
  await userEvent.clear(id);
  await userEvent.type(id, "my-python-docs");
  await userEvent.click(screen.getByLabelText("选择 ConfigParser"));
  await userEvent.click(screen.getByRole("button", { name: "刷新状态" }));
  expect(id).toHaveValue("my-python-docs");
});

test("model failure can resume with a note without requiring a URL", async () => {
  const reply = vi.fn(async () => task);
  mount({ createDiscoveryTask: async () => ({ ...task, status: "failed", stop_reason: "agent_model_failed" }), replyDiscoveryTask: reply });
  await userEvent.type(screen.getByLabelText("产品（必填）"), "pandas");
  await userEvent.click(screen.getByRole("button", { name: "查找文档" }));
  await userEvent.type(await screen.findByLabelText("补充说明"), "继续查找");
  await userEvent.click(screen.getByRole("button", { name: "继续当前任务" }));
  await waitFor(() => expect(reply).toHaveBeenCalledWith(task.task_id, expect.objectContaining({ message: "继续查找", urls: [] })));
});
function mount(overrides: Parameters<typeof fakeApi>[0] = {}) {
  const api = fakeApi({ getDiscoveryCapabilities: async () => caps, listKnowledgeBases: async () => ({ knowledge_bases: [] }), watchDiscoveryEvents: async () => {}, ...overrides });
  return { api, ...render(<MemoryRouter initialEntries={["/discovery"]}><ApiProvider client={api}><AppRoutes /></ApiProvider></MemoryRouter>) };
}

test("requires a product and defaults a blank requirement to official documentation", async () => {
  const create = vi.fn(async () => task);
  mount({ createDiscoveryTask: create });
  await userEvent.type(screen.getByLabelText("文档需求（可选）"), "Configuration docs");
  await userEvent.click(screen.getByRole("button", { name: "查找文档" }));
  expect(create).not.toHaveBeenCalled();
  await userEvent.clear(screen.getByLabelText("文档需求（可选）"));
  await userEvent.type(screen.getByLabelText("产品（必填）"), "pandas");
  await userEvent.click(screen.getByRole("button", { name: "查找文档" }));
  await waitFor(() => expect(create).toHaveBeenCalledWith(expect.objectContaining({ product: "pandas", goal: "查找 pandas 的官方文档" })));
});

test("shows provisional official-source reasoning and its real supporting page", async () => {
  sessionStorage.setItem("docguard.discovery.v1", JSON.stringify({ taskId: task.task_id }));
  const assessed = { ...candidate, source_status: "agent_assessed_official", source_assessment: {
    method: "linked_pages_model_assessment", reason: "项目主页指向该文档入口", requires_user_confirmation: true,
    independent_ownership_verified: false,
    authority_pages: [{ url: "https://example.org/", content_sha256: "source-hash", linked_document_urls: [candidate.url] }],
  } };
  mount({ getDiscoveryTask: async () => ({ ...task, candidates: [assessed] }) });
  expect(await screen.findByText("Agent 判断为官方 · 待你确认")).toBeInTheDocument();
  expect(screen.getByText("项目主页指向该文档入口")).toBeInTheDocument();
  expect(screen.getByRole("link", { name: "https://example.org/" })).toHaveAttribute("href", "https://example.org/");
  expect(screen.getByText(/页面链接关系不能独立证明域名归属/)).toBeInTheDocument();
});

test("clarifies, approves only selected inspected candidates, and queries the imported knowledge base", async () => {
  const reply = vi.fn(async () => task);
  const approve = vi.fn(async () => ({ job_id: "import-1", task_id: task.task_id, knowledge_base_id: "python-docs", status: "queued", result: null, failure_type: null }));
  const query = vi.fn(async (): Promise<KnowledgeAnswer> => ({ action: "answer", mode: "extractive", answer: "[E01] Read INI files.", citations: [{ evidence_id: "E01", chunk_id: "chunk-1", url: candidate.url, text: "Read INI files.", product: "Python", version: "3", section_path: ["ConfigParser"] }] }));
  mount({ createDiscoveryTask: async () => ({ ...task, status: "awaiting_input", candidates: [], pending_question: "需要哪个版本？" }), replyDiscoveryTask: reply, approveDiscoveryTask: approve, queryKnowledgeBase: query });
  await userEvent.type(screen.getByLabelText("文档需求（可选）"), "Python 配置文档");
  await userEvent.type(screen.getByLabelText("产品（必填）"), "Python");
  await userEvent.click(screen.getByRole("button", { name: "查找文档" }));
  expect(await screen.findByText("需要哪个版本？")).toBeInTheDocument();
  await userEvent.type(screen.getByLabelText("补充说明"), "使用 Python 3");
  await userEvent.type(screen.getByLabelText("补充版本（可选）"), "3");
  await userEvent.click(screen.getByRole("button", { name: "提交补充" }));
  expect(await screen.findByText("ConfigParser")).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "确认导入" })).toBeDisabled();
  await userEvent.click(screen.getByLabelText("选择 ConfigParser"));
  await userEvent.clear(screen.getByLabelText("知识库标识"));
  await userEvent.type(screen.getByLabelText("知识库标识"), "python-docs");
  await userEvent.click(screen.getByRole("button", { name: "确认导入" }));
  await waitFor(() => expect(approve).toHaveBeenCalledWith(task.task_id, expect.objectContaining({ candidate_ids: ["c1"], revision: 2, knowledge_base_id: "python-docs" })));
  expect(reply).toHaveBeenCalledWith(task.task_id, expect.objectContaining({ message: "使用 Python 3", version: "3" }));
});

test("explains blocked document DNS with an actionable message", async () => {
  sessionStorage.setItem("docguard.discovery.v1", JSON.stringify({ taskId: task.task_id }));
  mount({ getDiscoveryTask: async () => ({ ...task, status: "failed", candidates: [], failures: [{ url: candidate.url, error_type: "FetchError", reason_code: "dns_non_public" }] }) });
  expect(await screen.findByText(/文档域名解析到了非公网地址/)).toBeInTheDocument();
});

test("restores a saved task after refresh and shows per-page failures and approved scope", async () => {
  sessionStorage.setItem("docguard.discovery.v1", JSON.stringify({ taskId: task.task_id }));
  mount({ getDiscoveryTask: async () => ({ ...task, status: "ready", failures: [{ url: "https://example.org/missing", error_type: "FetchError", reason_code: "source_inspection_failed" }], approved_manifest: { candidates: [candidate], knowledge_base_id: "python-docs" }, job: { job_id: "import-1", task_id: task.task_id, knowledge_base_id: "python-docs", status: "succeeded", result: { document_count: 1, chunk_count: 46 }, failure_type: null } }), queryKnowledgeBase: async () => ({ action: "answer", mode: "extractive", answer: "[E01] Read INI files.", citations: [{ evidence_id: "E01", chunk_id: "chunk-1", url: candidate.url, text: "Read INI files.", product: "Python", version: "3", section_path: ["ConfigParser"] }] }) });
  expect(await screen.findByText("导入完成")).toBeInTheDocument();
  expect(screen.getByText("FetchError")).toBeInTheDocument();
  expect(screen.getByRole("heading", { name: "已批准清单" })).toBeInTheDocument();
  await userEvent.click(screen.getByRole("link", { name: "开始提问" }));
  await userEvent.type(screen.getByLabelText("问题"), "How to read INI?");
  await userEvent.click(screen.getByRole("button", { name: "开始检索" }));
  await screen.findByRole("link", { name: "E01" });
  expect(document.getElementById("answer-text")).toHaveTextContent("[E01] Read INI files.");
  expect(screen.getByText("检索原文摘录")).toBeInTheDocument();
  await userEvent.click(screen.getByRole("link", { name: /^E01 · ConfigParser/ }));
  expect(screen.getByRole("link", { name: "打开文档原文" })).toHaveAttribute("href", candidate.url);
});

test("keeps in-flight creation and its result when switching pages", async () => {
  let resolve!: (value: DiscoveryTask) => void;
  const pending = new Promise<DiscoveryTask>(r => { resolve = r; });
  mount({ createDiscoveryTask: async () => pending });
  await userEvent.type(screen.getByLabelText("文档需求（可选）"), "Python 文档");
  await userEvent.type(screen.getByLabelText("产品（必填）"), "Python");
  await userEvent.click(screen.getByRole("button", { name: "查找文档" }));
  await userEvent.click(screen.getByRole("link", { name: /技术问答/ }));
  await act(async () => { resolve(task); });
  await userEvent.click(screen.getByRole("link", { name: /知识库/ }));
  await userEvent.click(screen.getByRole("link", { name: "添加技术文档" }));
  expect(screen.getByLabelText("文档需求（可选）")).toHaveValue("Python 文档");
  expect(await screen.findByText("ConfigParser")).toBeInTheDocument();
  expect(JSON.parse(sessionStorage.getItem("docguard.discovery.v1")!).taskId).toBe(task.task_id);
});

test("supports direct URLs when search is unavailable and never submits blank requests", async () => {
  const create = vi.fn(async () => task);
  mount({ getDiscoveryCapabilities: async () => ({ ...caps, agent_enabled: false, search_configured: false }), createDiscoveryTask: create });
  expect(await screen.findByText(/自动查找未启用/)).toBeInTheDocument();
  await userEvent.click(screen.getByRole("button", { name: "查找文档" }));
  expect(create).not.toHaveBeenCalled();
  await userEvent.type(screen.getByLabelText("文档需求（可选）"), "Python 配置文档");
  await userEvent.type(screen.getByLabelText("产品（必填）"), "Python");
  await userEvent.type(screen.getByLabelText("文档网址（必填，每行一个）"), candidate.url);
  await userEvent.click(screen.getByRole("button", { name: "查找文档" }));
  await waitFor(() => expect(create).toHaveBeenCalledWith(expect.objectContaining({ urls: [candidate.url] })));
});

test("a stale approval refreshes candidates, clears selection, and requires a fresh confirmation", async () => {
  sessionStorage.setItem("docguard.discovery.v1", JSON.stringify({ taskId: task.task_id }));
  let revised = false;
  mount({ getDiscoveryTask: async () => ({ ...task, revision: revised ? 3 : 2 }), approveDiscoveryTask: async () => { revised = true; throw new ApiError(409, "state_conflict", "conflict"); } });
  await screen.findByText("ConfigParser");
  await userEvent.click(screen.getByLabelText("选择 ConfigParser"));
  await userEvent.clear(screen.getByLabelText("知识库标识"));
  await userEvent.type(screen.getByLabelText("知识库标识"), "python-docs");
  await userEvent.click(screen.getByRole("button", { name: "确认导入" }));
  expect(await screen.findByText(/候选修订或任务状态已变化/)).toBeInTheDocument();
  expect(await screen.findByText("修订 3")).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "确认导入" })).toBeDisabled();
  expect(screen.getByLabelText("选择 ConfigParser")).not.toBeChecked();
});

test("reconnects an interrupted event stream from its last event and keeps polling without duplicating events", async () => {
  sessionStorage.setItem("docguard.discovery.v1", JSON.stringify({ taskId: task.task_id }));
  const read = vi.fn(async () => ({ ...task, status: "discovering" }));
  const watch = vi.fn(async (_id, _after, onEvent) => { onEvent({ event_id: 8, kind: "page_read", data: { url: candidate.url } }); throw new Error("disconnected"); });
  mount({ getDiscoveryTask: read, watchDiscoveryEvents: watch, cancelDiscoveryTask: async () => ({ ...task, status: "cancelled" }) });
  await waitFor(() => expect(watch).toHaveBeenCalledTimes(2), { timeout: 4000 });
  expect(watch.mock.calls[1][1]).toBe(8);
  expect(screen.getAllByText("#8 page_read")).toHaveLength(1);
  expect(read.mock.calls.length).toBeGreaterThan(2);
  await userEvent.click(screen.getByRole("button", { name: "取消任务" }));
  expect(await screen.findByText("已取消")).toBeInTheDocument();
});

test("pending knowledge queries and completed answers survive page changes", async () => {
  sessionStorage.setItem("docguard.discovery.v1", JSON.stringify({ taskId: task.task_id }));
  let resolve!: (answer: KnowledgeAnswer) => void;
  const pending = new Promise<KnowledgeAnswer>(r => { resolve = r; });
  mount({ getDiscoveryTask: async () => ({ ...task, status: "ready", job: { job_id: "i1", task_id: task.task_id, knowledge_base_id: "python-docs", status: "succeeded", result: null, failure_type: null } }), queryKnowledgeBase: async () => pending });
  await screen.findByText("导入完成");
  await userEvent.click(screen.getByRole("link", { name: "开始提问" }));
  await userEvent.type(screen.getByLabelText("问题"), "read ini");
  await userEvent.click(screen.getByRole("button", { name: "开始检索" }));
  await userEvent.click(screen.getByRole("link", { name: /知识库/ }));
  await userEvent.click(screen.getByRole("link", { name: /技术问答/ }));
  expect(screen.getByRole("button", { name: "开始检索" })).toBeDisabled();
  await act(async () => resolve({ action: "answer", mode: "extractive", answer: "Persisted answer", citations: [] }));
  expect(await screen.findByText("Persisted answer")).toBeInTheDocument();
});

test("replays the event ledger for a restored completed task", async () => {
  sessionStorage.setItem("docguard.discovery.v1", JSON.stringify({ taskId: task.task_id }));
  const watch = vi.fn(async (_id, _after, onEvent) => { onEvent({ event_id: 9, kind: "import_completed", data: { status: "ready" } }); });
  mount({ getDiscoveryTask: async () => ({ ...task, status: "ready" }), watchDiscoveryEvents: watch });
  expect(await screen.findByText("#9 import_completed")).toBeInTheDocument();
});

test("a budget-exhausted task is terminal and allows creating a new task", async () => {
  sessionStorage.setItem("docguard.discovery.v1", JSON.stringify({ taskId: task.task_id }));
  mount({ getDiscoveryTask: async () => ({ ...task, status: "exhausted", candidates: [], stop_reason: "agent_budget_exhausted" }) });
  expect(await screen.findByText("预算耗尽")).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "查找文档" })).toBeEnabled();
  expect(screen.queryByRole("button", { name: "取消任务" })).not.toBeInTheDocument();
});


test("document presets are sent and collection approval expands the selected entry", async () => {
  const create = vi.fn(async () => ({ ...task, candidates: [{ ...candidate, collection: { root_url: candidate.url, path_prefix: "/guide/", max_pages: 1000 } }] }));
  const approve = vi.fn(async () => ({ job_id: "import-collection", task_id: task.task_id, knowledge_base_id: "python-configparser", status: "queued", result: null, failure_type: null }));
  mount({ createDiscoveryTask: create, approveDiscoveryTask: approve });
  expect(screen.getByLabelText("文档类型")).toHaveValue("user_guide");
  await userEvent.selectOptions(screen.getByLabelText("文档类型"), "api_reference");
  await userEvent.type(screen.getByLabelText("产品（必填）"), "Python");
  await userEvent.click(screen.getByRole("button", { name: "查找文档" }));
  await screen.findByLabelText("知识库标识");
  expect(create).toHaveBeenCalledWith(expect.objectContaining({ document_type: "api_reference" }));
  await userEvent.click(screen.getByLabelText("选择 ConfigParser"));
  await userEvent.click(screen.getByRole("button", { name: "确认导入" }));
  await waitFor(() => expect(approve).toHaveBeenCalledWith(task.task_id, expect.objectContaining({ import_scope: "collection" })));
});


test("failed imports can retry the approved task without another discovery", async () => {
  const failedJob = { job_id: "import-failed", task_id: task.task_id, knowledge_base_id: "python-docs", status: "failed", result: null, failure_type: "ChunkingError" };
  const retry = vi.fn(async () => ({ ...failedJob, status: "queued", failure_type: null }));
  sessionStorage.setItem("docguard.discovery.v1", JSON.stringify({ taskId: task.task_id }));
  mount({ getDiscoveryTask: async () => ({ ...task, status: "failed", job: failedJob }), retryImport: retry });
  await userEvent.click(await screen.findByRole("button", { name: "重试导入（复用已下载页面）" }));
  await waitFor(() => expect(retry).toHaveBeenCalledWith("import-failed"));
});


test.each([
  ["failed", "agent_model_failed"],
  ["failed", "no_verified_candidates"],
  ["exhausted", "agent_budget_exhausted"],
  ["awaiting_input", "search_network_failed"],
  ["awaiting_input", "dns_resolution_failed"],
])("%s search can resubmit original parameters after restore without cancelling or adding a URL", async (status, stop_reason) => {
  const previous = { ...task, status, stop_reason, product: "pandas", version: "2.3", goal: "Find pandas API docs", urls: [], official_only: false, document_type: "api_reference" };
  const create = vi.fn(async () => ({ ...previous, task_id: "discover-new", status: "discovering", stop_reason: undefined }));
  const cancel = vi.fn();
  sessionStorage.setItem("docguard.discovery.v1", JSON.stringify({ taskId: previous.task_id }));
  mount({ getDiscoveryTask: async id => id === previous.task_id ? previous : { ...previous, task_id: "discover-new", status: "discovering" }, createDiscoveryTask: create, cancelDiscoveryTask: cancel });
  await userEvent.click(await screen.findByRole("button", { name: "重新提交搜索" }));
  await waitFor(() => expect(create).toHaveBeenCalledWith({ product: "pandas", version: "2.3", goal: "Find pandas API docs", urls: [], official_only: false, document_type: "api_reference" }));
  expect(cancel).not.toHaveBeenCalled();
  expect(await screen.findByText("discover-new")).toBeInTheDocument();
});

test("resubmitting explicit failed URLs preserves them and active searches cannot resubmit", async () => {
  const previous = { ...task, status: "failed", urls: [candidate.url], document_type: "topic" };
  const create = vi.fn(async () => ({ ...previous, task_id: "discover-new", status: "discovering" }));
  sessionStorage.setItem("docguard.discovery.v1", JSON.stringify({ taskId: previous.task_id }));
  mount({ getDiscoveryTask: async id => id === previous.task_id ? previous : { ...previous, task_id: "discover-new", status: "discovering" }, createDiscoveryTask: create });
  await userEvent.click(await screen.findByRole("button", { name: "重新提交搜索" }));
  await waitFor(() => expect(create).toHaveBeenCalledWith(expect.objectContaining({ urls: [candidate.url], document_type: "topic" })));
  await screen.findByText("discover-new");
  expect(screen.queryByRole("button", { name: "重新提交搜索" })).not.toBeInTheDocument();
});


test("import progress is prominently placed before both discovery panels", async () => {
  const job = { job_id: "import-progress", task_id: task.task_id, knowledge_base_id: "python-docs", status: "running", result: null, failure_type: null, progress: { phase: "embedding", completed: 32, total: 40, unit: "chunks" } };
  sessionStorage.setItem("docguard.discovery.v1", JSON.stringify({ taskId: task.task_id }));
  mount({ getDiscoveryTask: async () => ({ ...task, status: "importing", job }) });
  const progress = await screen.findByRole("region", { name: "导入进度" });
  expect(within(progress).getByText("32 / 40 个分块")).toBeVisible();
  expect(progress.compareDocumentPosition(screen.getByRole("region", { name: "需求与补充" })) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
});
