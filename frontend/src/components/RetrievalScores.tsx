import type { ProvenanceResponse } from "../api/types";

export function RetrievalScores({ provenance }: { provenance?: ProvenanceResponse }) {
  if (!provenance) return <p className="muted">本次响应未提供检索追踪。</p>;
  const rank = (value: number | null) => value !== null && Number.isInteger(value) && value > 0 ? value : "—";
  const score = (value: number | null) => value !== null && Number.isFinite(value) ? value.toFixed(4) : "—";
  return <div className="score-grid" aria-label="检索追踪">
    <div className="score-card"><strong>Dense #{rank(provenance.dense_rank)}</strong></div>
    <div className="score-card"><strong>BM25 #{rank(provenance.bm25_rank)}</strong></div>
    <div className="score-card"><small>RRF</small><strong>{score(provenance.rrf_score)}</strong></div>
    <div className="score-card"><small>Rerank</small><strong>{score(provenance.rerank_score)}</strong></div>
  </div>;
}
