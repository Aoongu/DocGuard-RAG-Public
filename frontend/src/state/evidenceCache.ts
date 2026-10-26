import type { EvidenceSummary } from "../api/types";

const KEY = "docguard.query-evidence.v1";
export interface CachedEvidence { evidence: EvidenceSummary; used: boolean; }

function readAll(): Record<string, CachedEvidence> {
  try { return JSON.parse(sessionStorage.getItem(KEY) ?? "{}") as Record<string, CachedEvidence>; } catch { return {}; }
}
export function cacheEvidence(evidence: EvidenceSummary, used: boolean) { sessionStorage.setItem(KEY, JSON.stringify({ ...readAll(), [evidence.evidence_id]: { evidence, used } })); }
export function getCachedEvidence(id: string): CachedEvidence | null { return readAll()[id] ?? null; }
