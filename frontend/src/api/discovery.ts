import type { ProvenanceResponse } from "./types";

export interface DiscoveryCandidate {
  collection?: { root_url: string; path_prefix: string; max_pages: number } | null;
  candidate_id: string;
  url: string;
  title: string;
  content_sha256: string;
  version: string | null;
  version_status: string;
  source_status: string;
  source_assessment?: {
    method: string;
    reason: string;
    requires_user_confirmation: boolean;
    independent_ownership_verified: boolean;
    authority_pages: { url: string; content_sha256: string; linked_document_urls: string[] }[];
  };
  status: string;
}
export interface ImportJob {
  job_id: string;
  task_id: string;
  knowledge_base_id: string;
  status: string;
  result: { document_count?: number; chunk_count?: number; snapshot_id?: string } | null;
  failure_type: string | null;
  failure_reason?: string;
  failure_url?: string | null;
  progress?: { phase: string; completed_pages?: number; discovered_pages?: number; completed?: number; total?: number; unit?: string; current_url?: string; last_url?: string; chunk_count?: number; updated_at?: string };
}
export interface DiscoveryTask {
  urls?: string[];
  official_only?: boolean;
  document_type?: string;
  task_id: string;
  goal: string;
  product: string;
  version: string | null;
  status: string;
  revision: number;
  candidates: DiscoveryCandidate[];
  failures: { url: string; error_type: string; reason_code: string; network_attempts?: { address: string; error_type: string; phase: string; elapsed_ms: number }[] }[];
  pending_question: string | null;
  stop_reason?: string;
  job: ImportJob | null;
  approved_manifest?: { candidates: DiscoveryCandidate[]; knowledge_base_id: string };
}
export interface DiscoveryCreate {
  document_type?: string;
  goal?: string;
  product: string;
  version?: string | null;
  urls: string[];
  official_only: boolean;
}
export interface DiscoveryReply { message?: string; urls?: string[]; version?: string; product?: string; allow_unversioned?: boolean }
export interface DiscoveryApproval { import_scope?: "pages" | "collection"; candidate_ids: string[]; revision: number; knowledge_base_id: string; idempotency_key: string }
export interface DiscoveryCapabilities { html_import: boolean; decision_model_configured: boolean; search_configured: boolean; agent_enabled: boolean; search_provider?: string; document_dns_mode?: string }
export interface KnowledgeBase { knowledge_base_id: string; product: string; version: string | null; document_count: number; chunk_count: number; snapshot_id: string }
export interface KnowledgeAnswer {
  action: "answer" | "refuse" | "clarify";
  mode: "extractive" | "generated";
  answer: string | null;
  reason?: string;
  explanation?: string | null;
  evidence?: KnowledgeAnswer["citations"];
  decision?: { action: string; reason_code: string | null; confidence: number | null; triggered_rule: string };
  model_info?: { model: string; request_id: string; input_tokens: number; output_tokens: number; cached_input_tokens?: number } | null;
  retrieval?: { method: string; snapshot_id: string; embedding_id: string | null; reranker_id: string | null; rrf_k: number | null };
  refusal_calibration?: string;
  citations: { evidence_id: string; chunk_id: string; url: string; text: string; product: string; version: string | null; section_path: string[]; rank?: number; score?: number; provenance?: ProvenanceResponse }[];
}
export interface DiscoveryEvent { event_id: number; kind: string; data: Record<string, unknown> }

/** Reject unsafe source links even if a remote or restored payload was malformed. */
export function sourceUrl(value: string): string | undefined {
  try { const url = new URL(value); return url.protocol === "https:" && !url.username && !url.password ? url.href : undefined; } catch { return undefined; }
}

export async function watchEvents(fetcher: typeof fetch, taskId: string, after: number, onEvent: (event: DiscoveryEvent) => void, signal?: AbortSignal): Promise<void> {
  const response = await fetcher(`/api/v1/discovery/tasks/${encodeURIComponent(taskId)}/events`, {
    headers: { Accept: "text/event-stream", "Last-Event-ID": String(after) }, signal,
  });
  if (!response.ok || !response.headers.get("Content-Type")?.includes("text/event-stream") || !response.body) throw new Error(`事件连接不可用（HTTP ${response.status}），正在轮询任务状态`);
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "", cursor = after;
  function consume() {
    let match: RegExpExecArray | null;
    while ((match = /\r?\n\r?\n/.exec(buffer))) {
      const frame = buffer.slice(0, match.index); buffer = buffer.slice(match.index + match[0].length);
      let id = "", kind = "message"; const data: string[] = [];
      for (const line of frame.split(/\r?\n/)) {
        const colon = line.indexOf(":"); if (colon < 1) continue;
        const field = line.slice(0, colon), value = line.slice(colon + 1).replace(/^ /, "");
        if (field === "id") id = value;
        else if (field === "event") kind = value;
        else if (field === "data") data.push(value);
      }
      const eventId = Number(id);
      if (!/^\d+$/.test(id) || !Number.isSafeInteger(eventId) || eventId <= cursor || !data.length) continue;
      const payload: unknown = JSON.parse(data.join("\n"));
      if (!payload || typeof payload !== "object" || Array.isArray(payload)) throw new Error("事件内容格式无效");
      onEvent({ event_id: eventId, kind, data: payload as Record<string, unknown> }); cursor = eventId;
    }
    if (buffer.length > 1_000_000) throw new Error("事件长度超限");
  }
  try {
    while (true) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value, { stream: !done }); consume();
      if (done) break;
    }
  } finally { await reader.cancel().catch(() => {}); reader.releaseLock(); }
}


/** Resubmission creates a separate run; importing tasks keep their approved job. */
export function canResubmitDiscovery(task: DiscoveryTask | null): boolean {
  return !!task && !task.job && (["failed", "exhausted", "cancelled"].includes(task.status) || (task.status === "awaiting_input" && !!task.stop_reason));
}
