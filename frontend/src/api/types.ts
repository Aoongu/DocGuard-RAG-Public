export type PostgreSqlVersion = "17.11" | "18.6";
export type DecisionAction = "answer" | "refuse" | "clarify" | "error";
export type DecisionReason =
  | "OUT_OF_DOMAIN"
  | "NO_RELEVANT_EVIDENCE"
  | "VERSION_NOT_AVAILABLE"
  | "VERSION_AMBIGUOUS"
  | "VERSION_CONFLICT"
  | "EVIDENCE_INSUFFICIENT"
  | "CITATION_VALIDATION_FAILED"
  | "UPSTREAM_ERROR";

export interface QueryRequest {
  question: string;
  requested_version: PostgreSqlVersion | null;
}

export interface DecisionResponse {
  action: DecisionAction;
  reason_code: DecisionReason | null;
  confidence: number;
  triggered_rule: string;
  feature_values: Record<string, unknown>;
}

export interface ProvenanceResponse {
  sources: string[];
  dense_rank: number | null;
  dense_score: number | null;
  bm25_rank: number | null;
  bm25_score: number | null;
  rrf_score: number | null;
  rerank_score: number | null;
  rerank_truncated: boolean | null;
  original_rrf_rank: number | null;
}

export interface EvidenceSummary {
  evidence_id: string;
  chunk_id: string;
  rank: number;
  score: number;
  postgresql_version: PostgreSqlVersion;
  topic: string;
  section_path: string[];
  canonical_url: string;
  anchor: string;
  text: string;
  provenance: ProvenanceResponse;
}

export interface CitationResponse {
  evidence_id: string;
  chunk_id: string;
  postgresql_version: PostgreSqlVersion;
  topic: string;
  section_path: string[];
  canonical_url: string;
  anchor: string;
  text: string;
  score: number;
}

export interface ModelInfoResponse {
  model: string;
  request_id: string | null;
  provider_fingerprint: string | null;
  input_tokens: number;
  cached_input_tokens: number;
  output_tokens: number;
  api_cost_usd: number;
}

export interface QueryResponse {
  request_id: string;
  question: string;
  requested_version: PostgreSqlVersion | null;
  route: "m3" | "m4";
  decision: DecisionResponse;
  answer: string | null;
  citations: CitationResponse[];
  evidence: EvidenceSummary[];
  latency_ms: Record<string, number>;
  model_info: ModelInfoResponse | null;
}

export interface DependencyHealth {
  name: string;
  status: "available" | "unavailable";
  detail: string | null;
}

export interface HealthResponse {
  request_id: string;
  status: "ok" | "degraded";
  dependencies: DependencyHealth[];
}

export interface CorpusResponse {
  version: PostgreSqlVersion;
  document_count: number;
  chunk_count: number;
  index_status: "ready" | "unavailable";
  snapshot_sha256: string | null;
  built_at: string | null;
}

export interface CorpusListResponse {
  request_id: string;
  corpora: CorpusResponse[];
}

export interface RebuildResponse {
  request_id: string;
  job_id: string;
  status: "queued";
}

export interface EvidenceResponse {
  request_id: string;
  chunk_id: string;
  postgresql_version: PostgreSqlVersion;
  topic: string;
  section_path: string[];
  canonical_url: string;
  anchor: string;
  text: string;
  content_sha256: string;
  chunking_version: string;
}

export type EvaluationMethod = "b0" | "b1" | "m1" | "m2" | "b2" | "m4";
export type EvaluationSplit = "development" | "validation";
export type EvaluationStatus = "queued" | "running" | "succeeded" | "failed";

export interface EvaluationCreateRequest {
  method: EvaluationMethod;
  split: EvaluationSplit;
}

export interface EvaluationJobResponse {
  run_id: string;
  method: EvaluationMethod;
  split: EvaluationSplit;
  status: EvaluationStatus;
  created_at: string;
  updated_at: string;
  metrics: Record<string, unknown> | null;
  failure_type: string | null;
  failure_message: string | null;
}

export interface EvaluationListResponse {
  request_id: string;
  evaluations: EvaluationJobResponse[];
}

export interface ApiProblem {
  request_id: string;
  error: {
    type: string;
    message: string;
  };
}
