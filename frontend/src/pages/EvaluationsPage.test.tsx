import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { expect, test } from "vitest";
import { ApiProvider } from "../api/context";
import { fakeApi } from "../test/fakeApi";
import { EvaluationsPage } from "./EvaluationsPage";

test("does not manufacture absent metric values", async () => {
  const api = fakeApi({ listEvaluations: async () => ({ request_id: "r", evaluations: [{ run_id: "run-1", method: "m4", split: "validation", status: "succeeded", created_at: "2026-10-21", updated_at: "2026-10-21", metrics: { "Recall@10": .82 }, failure_type: null, failure_message: null }] }) });
  render(<ApiProvider client={api}><EvaluationsPage /></ApiProvider>);
  await userEvent.selectOptions(screen.getByLabelText("方法"), "m4");
  expect(await screen.findByText("Recall@10")).toBeInTheDocument();
  expect(screen.queryByText("nDCG@10")).not.toBeInTheDocument();
});

test("shows the empty state returned by the API", async () => {
  const api = fakeApi({ listEvaluations: async () => ({ request_id: "r", evaluations: [] }) });
  render(<ApiProvider client={api}><EvaluationsPage /></ApiProvider>);
  expect(await screen.findByText("暂无评测运行")).toBeInTheDocument();
});

test("filters jobs by the exact selected method", async () => {
  const api = fakeApi({ listEvaluations: async () => ({ request_id: "r", evaluations: [
    { run_id: "run-b0", method: "b0", split: "development", status: "failed", created_at: "2026-10-23", updated_at: "2026-10-23", metrics: null, failure_type: "RuntimeError", failure_message: "old failure" },
    { run_id: "run-m4", method: "m4", split: "development", status: "failed", created_at: "2026-10-23", updated_at: "2026-10-23", metrics: null, failure_type: "RuntimeError", failure_message: "protocol locked" },
  ] }) });
  render(<ApiProvider client={api}><EvaluationsPage /></ApiProvider>);

  expect(await screen.findByText("run-b0")).toBeInTheDocument();
  expect(screen.queryByText("run-m4")).not.toBeInTheDocument();

  await userEvent.selectOptions(screen.getByLabelText("方法"), "m4");
  expect(screen.getByText("run-m4")).toBeInTheDocument();
  expect(screen.queryByText("run-b0")).not.toBeInTheDocument();

  await userEvent.selectOptions(screen.getByLabelText("方法"), "b1");
  expect(screen.getByText("该方法暂无评测运行")).toBeInTheDocument();
});

test("does not offer online creation for protocol-locked methods", async () => {
  const api = fakeApi({ listEvaluations: async () => ({ request_id: "r", evaluations: [] }) });
  render(<ApiProvider client={api}><EvaluationsPage /></ApiProvider>);

  await userEvent.selectOptions(screen.getByLabelText("方法"), "m4");
  expect(screen.getByRole("button", { name: "创建评测" })).toBeDisabled();
  expect(screen.getByText(/B2、M4/)).toBeInTheDocument();
});
