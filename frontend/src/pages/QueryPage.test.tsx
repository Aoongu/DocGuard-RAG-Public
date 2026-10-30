import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { expect, test, vi } from "vitest";
import { AppRoutes } from "../App";
import { ApiProvider } from "../api/context";
import type { QueryResponse } from "../api/types";
import { fakeApi } from "../test/fakeApi";
import { QuerySessionProvider } from "../state/querySession";
import { QueryPage } from "./QueryPage";

test("connects inline E3 and E03 to the actual third citation without renumbering", async () => {
  const citation = { evidence_id: "third-chunk", citation_label: "E03", chunk_id: "third-chunk", postgresql_version: "18.6" as const, topic: "vacuum", section_path: ["VACUUM"], canonical_url: "https://www.postgresql.org/docs/18/sql-vacuum.html", anchor: "VACUUM", text: "vacuum evidence", score: .8 };
  const api = fakeApi({ listKnowledgeBases: async () => ({ knowledge_bases: [] }), query: async ({ question }) => ({ request_id: "r", question, requested_version: null, route: "m4", decision: { action: "answer", reason_code: null, confidence: .536, triggered_rule: "evidence_sufficient", feature_values: {} }, answer: "First statement [E3]; second statement [E03].", citations: [citation], evidence: [], latency_ms: {}, model_info: null }) });
  render(<MemoryRouter><ApiProvider client={api}><QuerySessionProvider><QueryPage /></QuerySessionProvider></ApiProvider></MemoryRouter>);
  await userEvent.type(screen.getByLabelText("问题"), "VACUUM options?");
  await userEvent.click(screen.getByRole("button", { name: "开始检索" }));
  const markers = await screen.findAllByRole("link", { name: "E03" });
  expect(markers).toHaveLength(2);
  expect(markers[0]).toHaveAttribute("href", "#citation-E03");
  await userEvent.click(markers[0]);
  expect(document.activeElement).toHaveAttribute("id", "citation-E03");
  expect(screen.getByRole("link", { name: "返回回答" })).toHaveAttribute("href", "#answer-text");
});

test("requests model generation for imported documents and shows truthful backend decision", async () => {
  const query = vi.fn(async () => ({ action: "answer" as const, mode: "generated" as const, answer: "Use ConfigParser [E01].", citations: [{ evidence_id: "E01", chunk_id: "py1", url: "https://docs.python.org/3/library/configparser.html", text: "INI source", product: "Python", version: "3", section_path: ["ConfigParser"], provenance: { sources: ["dense", "bm25"], dense_rank: 39, dense_score: .7, bm25_rank: 5, bm25_score: 2.1, rrf_score: .0569, rerank_score: -.3432, rerank_truncated: false, original_rrf_rank: 2 } }], retrieval: { method: "full", snapshot_id: "snapshot", embedding_id: "e5", reranker_id: "bge", rrf_k: 20 }, decision: { action: "answer", triggered_rule: "generic_citations_validated", reason_code: null, confidence: null }, model_info: { model: "mock-deepseek", request_id: "r-model", input_tokens: 120, output_tokens: 24 } }));
  const api = fakeApi({ listKnowledgeBases: async () => ({ knowledge_bases: [] }), queryKnowledgeBase: query });
  render(<MemoryRouter initialEntries={["/query?kb=python-docs"]}><ApiProvider client={api}><QuerySessionProvider><QueryPage /></QuerySessionProvider></ApiProvider></MemoryRouter>);
  await userEvent.type(screen.getByLabelText("问题"), "Read INI files?");
  await userEvent.click(screen.getByRole("button", { name: "开始检索" }));
  expect(await screen.findByText("BACKEND DECISION")).toBeInTheDocument();
  expect(screen.getByText(/规则：generic_citations_validated.*置信度.*未校准/)).toBeInTheDocument();
  expect(screen.getByText(/mock-deepseek/)).toBeInTheDocument();
  expect(query).toHaveBeenCalledWith("python-docs", "Read INI files?", expect.objectContaining({ generate: true }));
  expect(screen.getByRole("link", { name: "E01" })).toHaveAttribute("href", "#citation-E01");
  expect(screen.queryByText("Dense #39")).not.toBeInTheDocument();
  expect(screen.queryByText("BM25 #5")).not.toBeInTheDocument();
  expect(screen.queryByText("0.0569")).not.toBeInTheDocument();
  expect(screen.getByRole("link", { name: /E01 · ConfigParser/ })).toHaveAttribute("href", "/evidence?kb=python-docs&chunk=py1");
});

test("shows imported-document refusal as REFUSE and preserves its explanation", async () => {
  const api = fakeApi({ listKnowledgeBases: async () => ({ knowledge_bases: [] }), queryKnowledgeBase: async () => ({ action: "refuse", mode: "generated", answer: null, explanation: "当前 ConfigParser 文档没有整数缓存资料。", citations: [], evidence: [{ evidence_id: "E01", chunk_id: "refusal-py1", url: "https://docs.python.org/3/library/configparser.html", text: "INI source", product: "Python", version: "3", section_path: ["ConfigParser"] }], decision: { action: "refuse", reason_code: "EVIDENCE_INSUFFICIENT", confidence: null, triggered_rule: "generic_model_evidence_insufficient" } }) });
  render(<MemoryRouter initialEntries={["/query?kb=python-docs"]}><ApiProvider client={api}><QuerySessionProvider><QueryPage /></QuerySessionProvider></ApiProvider></MemoryRouter>);
  await userEvent.type(screen.getByLabelText("问题"), "is 与 == 的区别？");
  await userEvent.click(screen.getByRole("button", { name: "开始检索" }));
  expect(await screen.findByText("REFUSE")).toBeInTheDocument();
  expect(screen.getByText("当前 ConfigParser 文档没有整数缓存资料。")).toBeInTheDocument();
  expect(screen.queryByText("ANSWER")).not.toBeInTheDocument();
  await userEvent.click(screen.getByText("查看未用于作答的检索候选"));
  expect(screen.getByRole("link", { name: "检索候选 E01 · ConfigParser" })).toHaveAttribute("href", "/evidence?kb=python-docs&chunk=refusal-py1");
});

test("renders the backend refusal reason verbatim", async () => {
  const api = fakeApi({ query: async ({ question }) => ({ request_id: "r", question, requested_version: null, route: "m4", decision: { action: "refuse", reason_code: "EVIDENCE_INSUFFICIENT", confidence: .8, triggered_rule: "coverage", feature_values: {} }, answer: null, citations: [], evidence: [], latency_ms: { total: 42 }, model_info: null }) });
  render(<MemoryRouter><ApiProvider client={api}><QuerySessionProvider><QueryPage /></QuerySessionProvider></ApiProvider></MemoryRouter>);
  await userEvent.type(screen.getByLabelText("问题"), "如何配置？");
  await userEvent.click(screen.getByRole("button", { name: "开始检索" }));
  expect(await screen.findByText("EVIDENCE_INSUFFICIENT")).toBeInTheDocument();
  expect(screen.queryByTestId("generated-answer")).not.toBeInTheDocument();
});

test("rejects a blank question without calling the API", async () => {
  render(<MemoryRouter><ApiProvider client={fakeApi()}><QuerySessionProvider><QueryPage /></QuerySessionProvider></ApiProvider></MemoryRouter>);
  await userEvent.click(screen.getByRole("button", { name: "开始检索" }));
  expect(screen.getByText("请输入问题")).toBeInTheDocument();
});

test("keeps the pending question and result across route changes", async () => {
  let resolveQuery!: (response: QueryResponse) => void;
  const query = new Promise<QueryResponse>((resolve) => { resolveQuery = resolve; });
  const api = fakeApi({ query: async () => query });
  render(
    <MemoryRouter initialEntries={["/query"]}>
      <ApiProvider client={api}><AppRoutes /></ApiProvider>
    </MemoryRouter>,
  );

  await userEvent.type(screen.getByLabelText("问题"), "VACUUM 有哪些主要选项？");
  await userEvent.click(screen.getByRole("button", { name: "开始检索" }));
  expect(screen.getByText("正在检索、判断并生成…")).toBeInTheDocument();

  await userEvent.click(screen.getByRole("link", { name: /知识库/ }));
  await userEvent.click(screen.getByRole("link", { name: /技术问答/ }));
  expect(screen.getByLabelText("问题")).toHaveValue("VACUUM 有哪些主要选项？");
  expect(screen.getByText("正在检索、判断并生成…")).toBeInTheDocument();

  resolveQuery({
    request_id: "r-route",
    question: "VACUUM 有哪些主要选项？",
    requested_version: null,
    route: "m4",
    decision: { action: "answer", reason_code: null, confidence: .9, triggered_rule: "answer", feature_values: {} },
    answer: "VACUUM 支持多种选项。",
    citations: [],
    evidence: [],
    latency_ms: { total: 42 },
    model_info: null,
  });
  expect(await screen.findByText("VACUUM 支持多种选项。")).toBeInTheDocument();
});
