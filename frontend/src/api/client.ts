import type {
  ApiProblem,
  CorpusListResponse,
  EvaluationCreateRequest,
  EvaluationJobResponse,
  EvaluationListResponse,
  EvidenceResponse,
  HealthResponse,
  QueryRequest,
  QueryResponse,
  RebuildResponse,
} from "./types";

const API_PREFIX = "/api/v1";

type RequestOptions = {
  signal?: AbortSignal;
};

function isApiProblem(value: unknown): value is ApiProblem {
  if (!value || typeof value !== "object") return false;
  const record = value as Record<string, unknown>;
  const error = record.error;
  return (
    typeof record.request_id === "string" &&
    !!error &&
    typeof error === "object" &&
    typeof (error as Record<string, unknown>).type === "string" &&
    typeof (error as Record<string, unknown>).message === "string"
  );
}

export class ApiError extends Error {
  readonly status: number;
  readonly type: string;
  readonly requestId: string | null;

  constructor(status: number, type: string, message: string, requestId: string | null = null) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.type = type;
    this.requestId = requestId;
  }
}

export async function requestJson<T>(
  fetcher: typeof fetch,
  path: string,
  init: RequestInit = {},
): Promise<T> {
  const method = init.method?.toUpperCase() ?? "GET";
  const headers = new Headers(init.headers);
  headers.set("Accept", "application/json");
  if (method !== "GET" && method !== "HEAD") headers.set("Content-Type", "application/json");

  const response = await fetcher(path, { ...init, headers: Object.fromEntries(headers.entries()) });
  if (response.status === 204) return undefined as T;

  let payload: unknown = null;
  if (response.headers.get("Content-Type")?.includes("application/json")) {
    try {
      payload = await response.json();
    } catch {
      payload = null;
    }
  }

  if (!response.ok) {
    if (isApiProblem(payload)) {
      throw new ApiError(
        response.status,
        payload.error.type,
        payload.error.message,
        payload.request_id,
      );
    }
    throw new ApiError(response.status, "HTTP_ERROR", `请求失败（HTTP ${response.status}）`);
  }
  if (payload === null) {
    throw new ApiError(response.status, "INVALID_RESPONSE", "服务返回了无效的 JSON 响应");
  }
  return payload as T;
}

export interface ApiClient {
  getHealth(options?: RequestOptions): Promise<HealthResponse>;
  getCorpora(options?: RequestOptions): Promise<CorpusListResponse>;
  rebuildCorpora(options?: RequestOptions): Promise<RebuildResponse>;
  query(payload: QueryRequest, options?: RequestOptions): Promise<QueryResponse>;
  getEvidence(evidenceId: string, options?: RequestOptions): Promise<EvidenceResponse>;
  listEvaluations(options?: RequestOptions): Promise<EvaluationListResponse>;
  createEvaluation(
    payload: EvaluationCreateRequest,
    options?: RequestOptions,
  ): Promise<EvaluationJobResponse>;
  getEvaluation(runId: string, options?: RequestOptions): Promise<EvaluationJobResponse>;
}

export function createApiClient(fetcher: typeof fetch = fetch): ApiClient {
  const get = <T>(path: string, options?: RequestOptions) =>
    requestJson<T>(fetcher, path, { signal: options?.signal });
  const post = <T>(path: string, body: unknown, options?: RequestOptions) =>
    requestJson<T>(fetcher, path, {
      method: "POST",
      body: body === undefined ? undefined : JSON.stringify(body),
      signal: options?.signal,
    });

  return {
    getHealth: (options) => get(`${API_PREFIX}/health`, options),
    getCorpora: (options) => get(`${API_PREFIX}/corpora`, options),
    rebuildCorpora: (options) => post(`${API_PREFIX}/corpora/rebuild`, undefined, options),
    query: (payload, options) => post(`${API_PREFIX}/query`, payload, options),
    getEvidence: (evidenceId, options) =>
      get(`${API_PREFIX}/evidence/${encodeURIComponent(evidenceId)}`, options),
    listEvaluations: (options) => get(`${API_PREFIX}/evaluations`, options),
    createEvaluation: (payload, options) => post(`${API_PREFIX}/evaluations`, payload, options),
    getEvaluation: (runId, options) =>
      get(`${API_PREFIX}/evaluations/${encodeURIComponent(runId)}`, options),
  };
}

export const apiClient = createApiClient();
