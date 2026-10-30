import type { ApiClient } from "../api/client";

export function fakeApi(overrides: Partial<ApiClient> = {}): ApiClient {
  const missing = () => Promise.reject(new Error("unexpected API call"));
  return {
    getDiscoveryCapabilities: missing,
    createDiscoveryTask: missing,
    getDiscoveryTask: missing,
    replyDiscoveryTask: missing,
    approveDiscoveryTask: missing,
    cancelDiscoveryTask: missing,
    retryImport: missing,
    getImport: missing,
    listKnowledgeBases: missing,
    deleteKnowledgeBase: missing,
    queryKnowledgeBase: missing,
    watchDiscoveryEvents: missing,
    getHealth: missing,
    getCorpora: missing,
    rebuildCorpora: missing,
    query: missing,
    getEvidence: missing,
    listEvaluations: missing,
    createEvaluation: missing,
    getEvaluation: missing,
    ...overrides,
  } as ApiClient;
}
