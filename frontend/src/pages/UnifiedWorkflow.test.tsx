import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { expect, test, vi } from "vitest";
import { AppRoutes } from "../App";
import { ApiProvider } from "../api/context";
import { fakeApi } from "../test/fakeApi";
import type { KnowledgeAnswer } from "../api/discovery";
const candidate = { candidate_id: "c1", url: "https://docs.python.org/3/library/configparser.html", title: "ConfigParser", content_sha256: "abc", version: "3", version_status: "url_version", source_status: "official_domain", status: "inspected" };
const task = { task_id: "discover-test", goal: "Python 配置文档", product: "Python", version: "3", revision: 2, status: "awaiting_selection", candidates: [candidate], failures: [], pending_question: null, job: null };

const base = { knowledge_base_id: "python-docs", product: "Python", version: "3", document_count: 1, chunk_count: 46, snapshot_id: "snapshot-1" };

test("deleting a knowledge base clears the selected query target and ignores its pending answer", async () => {
  let finish!: (value: KnowledgeAnswer) => void;
  let deleted = false;
  mount("/query?kb=python-docs", {
    queryKnowledgeBase: () => new Promise(resolve => { finish = resolve; }),
    listKnowledgeBases: async () => ({ knowledge_bases: deleted ? [] : [base] }),
    deleteKnowledgeBase: async () => { deleted = true; return { knowledge_base_id: base.knowledge_base_id, status: "deleted" }; },
  });
  await userEvent.type(screen.getByLabelText("问题"), "read ini");
  await userEvent.click(screen.getByRole("button", { name: "开始检索" }));
  await userEvent.click(screen.getByRole("link", { name: /知识库/ }));
  await userEvent.click(await screen.findByRole("button", { name: "删除 python-docs" }));
  await userEvent.click(screen.getByRole("button", { name: "确认删除" }));
  await screen.findByText("已删除 python-docs");
  await userEvent.click(screen.getByRole("link", { name: /技术问答/ }));
  expect(screen.getByLabelText("查询知识库")).toHaveValue("");
  expect(screen.getByRole("button", { name: "开始检索" })).not.toBeDisabled();
  await act(async () => finish({ action: "answer", mode: "extractive", answer: "Deleted stale answer", citations: [] }));
  expect(screen.queryByText("Deleted stale answer")).not.toBeInTheDocument();
});
function mount(route: string, overrides: Parameters<typeof fakeApi>[0] = {}) {
  const api = fakeApi({ getHealth: async () => ({ request_id: "h", status: "ok", dependencies: [] }), getCorpora: async () => ({ request_id: "c", corpora: [] }), listKnowledgeBases: async () => ({ knowledge_bases: [base] }), getDiscoveryCapabilities: async () => ({ html_import: true, agent_enabled: false, search_configured: false, decision_model_configured: false }), watchDiscoveryEvents: async () => {}, ...overrides });
  render(<MemoryRouter initialEntries={[route]}><ApiProvider client={api}><AppRoutes /></ApiProvider></MemoryRouter>);
  return api;
}

test("updates a failed task URL without cancelling or creating another task", async () => {
  sessionStorage.setItem("docguard.discovery.v1", JSON.stringify({ taskId: task.task_id }));
  const reply = vi.fn(async () => ({ ...task, revision: 3 }));
  const create = vi.fn(); const cancel = vi.fn();
  mount("/discovery", { getDiscoveryTask: async () => ({ ...task, status: "failed", candidates: [], failures: [{ url: candidate.url, error_type: "FetchError", reason_code: "dns_non_public" }] }), replyDiscoveryTask: reply, createDiscoveryTask: create, cancelDiscoveryTask: cancel });
  await screen.findByText("任务失败");
  await userEvent.type(screen.getByLabelText("补充网址（每行一个）"), candidate.url);
  await userEvent.click(screen.getByRole("button", { name: "更新网址并重新核查" }));
  await waitFor(() => expect(reply).toHaveBeenCalledWith(task.task_id, expect.objectContaining({ urls: [candidate.url] })));
  expect(await screen.findByText("修订 3")).toBeInTheDocument();
  expect(create).not.toHaveBeenCalled(); expect(cancel).not.toHaveBeenCalled();
});

test("requires a URL before creating a task when automatic search is unconfigured", async () => {
  const create = vi.fn();
  mount("/discovery", { createDiscoveryTask: create });
  await screen.findByText("网址导入模式");
  await userEvent.type(screen.getByLabelText("文档需求（可选）"), "Python 文档");
  await userEvent.type(screen.getByLabelText("产品（必填）"), "Python");
  await userEvent.click(screen.getByRole("button", { name: "查找文档" }));
  expect(await screen.findByText("当前未配置自动搜索，请填写文档网址。")).toBeInTheDocument();
  expect(create).not.toHaveBeenCalled();
});

test("lists imported documents in knowledge management and queries them in the existing question and evidence pages", async () => {
  const query = vi.fn(async (): Promise<KnowledgeAnswer> => ({ action: "answer", mode: "extractive", answer: "INI answer", citations: [{ evidence_id: "E01", chunk_id: "chunk-1", url: candidate.url, text: "INI source evidence", product: "Python", version: "3", section_path: ["ConfigParser"] }] }));
  const oldQuery = vi.fn();
  mount("/knowledge", { queryKnowledgeBase: query, query: oldQuery });
  expect(await screen.findByText("python-docs")).toBeInTheDocument();
  expect(screen.getByRole("link", { name: "添加技术文档" })).toHaveAttribute("href", "/discovery");
  expect(screen.queryByRole("link", { name: /05/ })).not.toBeInTheDocument();
  await userEvent.click(screen.getByRole("link", { name: "向此知识库提问" }));
  expect(screen.getByLabelText("查询知识库")).toHaveValue("python-docs");
  await userEvent.type(screen.getByLabelText("问题"), "read ini");
  await userEvent.click(screen.getByRole("button", { name: "开始检索" }));
  expect(await screen.findByText("INI answer")).toBeInTheDocument();
  expect(query).toHaveBeenCalledWith("python-docs", "read ini", expect.objectContaining({ signal: expect.any(AbortSignal) }));
  expect(oldQuery).not.toHaveBeenCalled();
  await userEvent.click(screen.getByRole("link", { name: /^E01 · ConfigParser/ }));
  expect(await screen.findByText("INI source evidence")).toBeInTheDocument();
  expect(screen.getByRole("link", { name: "打开文档原文" })).toHaveAttribute("href", candidate.url);
});

test("keeps an imported knowledge query pending and preserves its result across navigation", async () => {
  let finish!: (value: KnowledgeAnswer) => void;
  const pending = new Promise<KnowledgeAnswer>(resolve => { finish = resolve; });
  mount("/query?kb=python-docs", { queryKnowledgeBase: async () => pending });
  await userEvent.type(screen.getByLabelText("问题"), "read ini");
  await userEvent.click(screen.getByRole("button", { name: "开始检索" }));
  await userEvent.click(screen.getByRole("link", { name: /知识库/ }));
  await userEvent.click(screen.getByRole("link", { name: /技术问答/ }));
  expect(screen.getByLabelText("查询知识库")).toHaveValue("python-docs");
  expect(screen.getByRole("button", { name: "开始检索" })).toBeDisabled();
  await act(async () => finish({ action: "answer", mode: "extractive", answer: "Recovered INI answer", citations: [] }));
  expect(await screen.findByText("Recovered INI answer")).toBeInTheDocument();
});
