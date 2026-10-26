import type { ApiClient } from "../api/client";

export function fakeApi(overrides: Partial<ApiClient> = {}): ApiClient {
  const missing = () => Promise.reject(new Error("unexpected API call"));
  return {
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
