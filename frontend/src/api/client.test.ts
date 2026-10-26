import { afterEach, describe, expect, test, vi } from "vitest";

import { ApiError, createApiClient, requestJson } from "./client";

const fetchMock = vi.fn<typeof fetch>();

afterEach(() => {
  fetchMock.mockReset();
});

describe("API client", () => {
  test("returns JSON responses and forwards an abort signal", async () => {
    const controller = new AbortController();
    fetchMock.mockResolvedValue(Response.json({ request_id: "req-1", corpora: [] }));
    const api = createApiClient(fetchMock);

    await expect(api.getCorpora({ signal: controller.signal })).resolves.toEqual({
      request_id: "req-1",
      corpora: [],
    });
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/v1/corpora",
      expect.objectContaining({ signal: controller.signal }),
    );
  });

  test("sends JSON bodies with explicit headers", async () => {
    fetchMock.mockResolvedValue(
      Response.json({ request_id: "req-2", job_id: "rebuild-1", status: "queued" }, { status: 202 }),
    );
    const api = createApiClient(fetchMock);

    await api.rebuildCorpora();

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/v1/corpora/rebuild",
      expect.objectContaining({ method: "POST" }),
    );
    const requestInit = fetchMock.mock.calls[0]?.[1];
    const headers = new Headers(requestInit?.headers);
    expect(headers.get("Accept")).toBe("application/json");
    expect(headers.get("Content-Type")).toBe("application/json");
  });

  test("throws the stable structured API problem", async () => {
    fetchMock.mockResolvedValue(
      Response.json(
        {
          request_id: "req-failed",
          error: { type: "UPSTREAM_ERROR", message: "required local artifact is unavailable" },
        },
        { status: 503 },
      ),
    );
    const api = createApiClient(fetchMock);

    const error = await api.getCorpora().catch((reason: unknown) => reason);

    expect(error).toBeInstanceOf(ApiError);
    expect(error).toMatchObject({
      status: 503,
      type: "UPSTREAM_ERROR",
      requestId: "req-failed",
      message: "required local artifact is unavailable",
    });
  });

  test("does not leak a non-JSON response body", async () => {
    fetchMock.mockResolvedValue(
      new Response("<html>private proxy error</html>", {
        status: 502,
        headers: { "Content-Type": "text/html" },
      }),
    );

    const error = await createApiClient(fetchMock).getHealth().catch((reason: unknown) => reason);

    expect(error).toMatchObject({ status: 502, type: "HTTP_ERROR", message: "请求失败（HTTP 502）" });
    expect(String(error)).not.toContain("private proxy error");
  });

  test("returns undefined for an empty 204 response", async () => {
    fetchMock.mockResolvedValue(new Response(null, { status: 204 }));

    await expect(requestJson<void>(fetchMock, "/empty")).resolves.toBeUndefined();
  });

  test("encodes evidence and evaluation identifiers", async () => {
    fetchMock
      .mockResolvedValueOnce(
        Response.json({
          request_id: "req-3",
          chunk_id: "chunk/id with space",
          postgresql_version: "18.6",
          topic: "sql",
          section_path: ["SQL"],
          canonical_url: "https://www.postgresql.org/docs/18/sql.html",
          anchor: "SQL",
          text: "text",
          content_sha256: "abc",
          chunking_version: "s2",
        }),
      )
      .mockResolvedValueOnce(
        Response.json({
          run_id: "run/one",
          method: "m4",
          split: "validation",
          status: "queued",
          created_at: "2026-10-21T00:00:00Z",
          updated_at: "2026-10-21T00:00:00Z",
          metrics: null,
          failure_type: null,
          failure_message: null,
        }),
      );
    const api = createApiClient(fetchMock);

    await api.getEvidence("chunk/id with space");
    await api.getEvaluation("run/one");

    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      "/api/v1/evidence/chunk%2Fid%20with%20space",
      expect.any(Object),
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      "/api/v1/evaluations/run%2Fone",
      expect.any(Object),
    );
  });
});
