import { render, screen } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { expect, test } from "vitest";
import { ApiProvider } from "../api/context";
import { fakeApi } from "../test/fakeApi";
import { cacheEvidence, cacheKnowledgeEvidence } from "../state/evidenceCache";
import { EvidencePage } from "./EvidencePage";

test("shows actual imported-document scores without inventing a missing BM25 rank", () => {
  cacheKnowledgeEvidence("python", { evidence_id: "E01", chunk_id: "py-trace", url: "https://docs.python.org/3/", text: "INI source", product: "Python", version: "3", section_path: ["ConfigParser"], provenance: { sources: ["dense"], dense_rank: 4, dense_score: .7, bm25_rank: null, bm25_score: null, rrf_score: .0417, rerank_score: -.3432, rerank_truncated: false, original_rrf_rank: 2 } });
  render(<MemoryRouter initialEntries={["/evidence?kb=python&chunk=py-trace"]}><ApiProvider client={fakeApi()}><EvidencePage /></ApiProvider></MemoryRouter>);
  expect(screen.getByText("Dense #4")).toBeInTheDocument();
  expect(screen.getByText("BM25 #—")).toBeInTheDocument();
  expect(screen.getByText("-0.3432")).toBeInTheDocument();
});

test("opens the canonical source and shows query provenance", async () => {
  cacheEvidence({ evidence_id: "chk-1", chunk_id: "chk-1", rank: 1, score: .9, postgresql_version: "18.6", topic: "vacuum", section_path: ["SQL", "VACUUM"], canonical_url: "https://www.postgresql.org/docs/18/sql-vacuum.html#SQL-VACUUM", anchor: "SQL-VACUUM", text: "cached", provenance: { sources: ["dense"], dense_rank: 2, dense_score: .7, bm25_rank: null, bm25_score: null, rrf_score: .05, rerank_score: .8, rerank_truncated: false, original_rrf_rank: 1 } }, true);
  const api = fakeApi({ getEvidence: async () => ({ request_id: "r", chunk_id: "chk-1", postgresql_version: "18.6", topic: "vacuum", section_path: ["SQL", "VACUUM"], canonical_url: "https://www.postgresql.org/docs/18/sql-vacuum.html#SQL-VACUUM", anchor: "SQL-VACUUM", text: "canonical text", content_sha256: "sha", chunking_version: "s2-v1" }) });
  render(<MemoryRouter initialEntries={["/evidence/chk-1"]}><ApiProvider client={api}><Routes><Route path="/evidence/:evidenceId" element={<EvidencePage />} /></Routes></ApiProvider></MemoryRouter>);
  expect(await screen.findByText("Dense #2")).toBeInTheDocument();
  expect(screen.getByRole("link", { name: "打开 PostgreSQL 官方原文" })).toHaveAttribute("href", "https://www.postgresql.org/docs/18/sql-vacuum.html#SQL-VACUUM");
  expect(screen.getByText("canonical text")).toBeInTheDocument();
});
