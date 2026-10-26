import { render, screen } from "@testing-library/react";
import { expect, test } from "vitest";
import { ApiProvider } from "../api/context";
import { fakeApi } from "../test/fakeApi";
import { KnowledgePage } from "./KnowledgePage";

test("renders backend corpus values without substitutions", async () => {
  const api = fakeApi({
    getCorpora: async () => ({ request_id: "r", corpora: [{ version: "17.11", document_count: 92, chunk_count: 7413, index_status: "ready", snapshot_sha256: "abc123", built_at: "2026-10-20T00:00:00Z" }] }),
    getHealth: async () => ({ request_id: "h", status: "ok", dependencies: [{ name: "qdrant", status: "available", detail: null }] }),
  });
  render(<ApiProvider client={api}><KnowledgePage /></ApiProvider>);
  expect(await screen.findByText("7,413")).toBeInTheDocument();
  expect(screen.getByText("17.11")).toBeInTheDocument();
  expect(screen.getByText("qdrant")).toBeInTheDocument();
});

test("shows an empty state", async () => {
  const api = fakeApi({ getCorpora: async () => ({ request_id: "r", corpora: [] }), getHealth: async () => ({ request_id: "h", status: "degraded", dependencies: [] }) });
  render(<ApiProvider client={api}><KnowledgePage /></ApiProvider>);
  expect(await screen.findByText("还没有可用语料")).toBeInTheDocument();
});
