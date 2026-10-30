import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { expect, test, vi } from "vitest";
import { ApiProvider } from "../api/context";
import { fakeApi } from "../test/fakeApi";
import { KnowledgePage } from "./KnowledgePage";
import { MemoryRouter } from "react-router-dom";

test("deletes an imported document only after confirmation and updates the list", async () => {
  const remove = vi.fn(async () => ({ knowledge_base_id: "python-docs", status: "deleted" as const }));
  const api = fakeApi({
    getCorpora: async () => ({ request_id: "r", corpora: [] }),
    getHealth: async () => ({ request_id: "h", status: "ok", dependencies: [] }),
    listKnowledgeBases: async () => ({ knowledge_bases: [{ knowledge_base_id: "python-docs", product: "Python", version: "3", document_count: 1, chunk_count: 46, snapshot_id: "snap" }] }),
    deleteKnowledgeBase: remove,
  });
  render(<MemoryRouter><ApiProvider client={api}><KnowledgePage /></ApiProvider></MemoryRouter>);
  await userEvent.click(await screen.findByRole("button", { name: "删除 python-docs" }));
  expect(remove).not.toHaveBeenCalled();
  await userEvent.click(screen.getByRole("button", { name: "取消删除" }));
  expect(remove).not.toHaveBeenCalled();
  await userEvent.click(screen.getByRole("button", { name: "删除 python-docs" }));
  await userEvent.click(screen.getByRole("button", { name: "确认删除" }));
  await waitFor(() => expect(remove).toHaveBeenCalledWith("python-docs"));
  expect(screen.queryByRole("heading", { name: "python-docs" })).not.toBeInTheDocument();
});

test("renders backend corpus values without substitutions", async () => {
  const api = fakeApi({
    getCorpora: async () => ({ request_id: "r", corpora: [{ version: "17.11", document_count: 92, chunk_count: 7413, index_status: "ready", snapshot_sha256: "abc123", built_at: "2026-10-20T00:00:00Z" }] }),
    getHealth: async () => ({ request_id: "h", status: "ok", dependencies: [{ name: "qdrant", status: "available", detail: null }] }),
  });
  render(<MemoryRouter><ApiProvider client={api}><KnowledgePage /></ApiProvider></MemoryRouter>);
  expect(await screen.findByText("7,413")).toBeInTheDocument();
  expect(screen.getByText("17.11")).toBeInTheDocument();
  expect(screen.getByText("qdrant")).toBeInTheDocument();
});

test("shows an empty state", async () => {
  const api = fakeApi({ getCorpora: async () => ({ request_id: "r", corpora: [] }), getHealth: async () => ({ request_id: "h", status: "degraded", dependencies: [] }) });
  render(<MemoryRouter><ApiProvider client={api}><KnowledgePage /></ApiProvider></MemoryRouter>);
  expect(await screen.findByText("还没有可用语料")).toBeInTheDocument();
});
