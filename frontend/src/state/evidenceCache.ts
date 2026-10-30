import type { EvidenceSummary } from "../api/types";
import type { KnowledgeAnswer } from "../api/discovery";

const KEY = "docguard.query-evidence.v1";
export interface CachedEvidence { evidence: EvidenceSummary; used: boolean; }

function readAll(): Record<string, CachedEvidence> {
  try { return JSON.parse(sessionStorage.getItem(KEY) ?? "{}") as Record<string, CachedEvidence>; } catch { return {}; }
}
export function cacheEvidence(evidence: EvidenceSummary, used: boolean) { sessionStorage.setItem(KEY, JSON.stringify({ ...readAll(), [evidence.evidence_id]: { evidence, used } })); }
export function getCachedEvidence(id: string): CachedEvidence | null { return readAll()[id] ?? null; }

type KnowledgeCitation = KnowledgeAnswer["citations"][number];
const knowledgeEvidence = new Map<string, KnowledgeCitation>();
export function cacheKnowledgeEvidence(kb: string, citation: KnowledgeCitation) {
  if (knowledgeEvidence.size >= 200) knowledgeEvidence.delete(knowledgeEvidence.keys().next().value!);
  knowledgeEvidence.set(JSON.stringify([kb, citation.chunk_id]), citation);
}
export function getKnowledgeEvidence(kb: string, chunk: string) { return knowledgeEvidence.get(JSON.stringify([kb, chunk])); }
