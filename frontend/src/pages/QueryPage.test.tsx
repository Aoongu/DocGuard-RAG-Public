import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { expect, test } from "vitest";
import { AppRoutes } from "../App";
import { ApiProvider } from "../api/context";
import type { QueryResponse } from "../api/types";
import { fakeApi } from "../test/fakeApi";
import { QuerySessionProvider } from "../state/querySession";
import { QueryPage } from "./QueryPage";

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
