import { expect, test, vi } from "vitest";
import { createApiClient } from "./client";

test("knowledge generation is an explicit opt-in request and can be disabled", async () => {
  const fetcher = vi.fn<typeof fetch>().mockImplementation(async () => Response.json({}));
  const api = createApiClient(fetcher);
  await api.queryKnowledgeBase("python-docs", "Read INI", { generate: true });
  expect(JSON.parse(String(fetcher.mock.calls[0][1]?.body))).toEqual({ question: "Read INI", generate: true });
  await api.queryKnowledgeBase("python-docs", "Read INI", { generate: false });
  expect(JSON.parse(String(fetcher.mock.calls[1][1]?.body))).toEqual({ question: "Read INI", generate: false });
});

test("discovery client encodes identifiers, preserves approval revision and idempotency key", async () => {
  const fetcher = vi.fn<typeof fetch>().mockResolvedValue(Response.json({}));
  const api = createApiClient(fetcher);
  await api.approveDiscoveryTask("discover/a", { candidate_ids: ["c1"], revision: 4, knowledge_base_id: "python", idempotency_key: "approve-1" });
  expect(fetcher).toHaveBeenCalledWith("/api/v1/discovery/tasks/discover%2Fa/approve", expect.objectContaining({ method: "POST", body: JSON.stringify({ candidate_ids: ["c1"], revision: 4, knowledge_base_id: "python", idempotency_key: "approve-1" }) }));
});

test("SSE resumes with Last-Event-ID, parses fragmented CRLF frames and deduplicates replay", async () => {
  const body = new ReadableStream<Uint8Array>({ start(controller) {
    const bytes = new TextEncoder().encode('id: 7\r\nevent: old\r\ndata: {}\r\n\r\nid: 8\r\nevent: agent_tool\r\ndata: {"message":"页面已读取"}\r\n\r\n');
    controller.enqueue(bytes.slice(0, 91)); controller.enqueue(bytes.slice(91)); controller.close();
  } });
  const fetcher = vi.fn<typeof fetch>().mockResolvedValue(new Response(body, { headers: { "Content-Type": "text/event-stream" } }));
  const onEvent = vi.fn();
  await createApiClient(fetcher).watchDiscoveryEvents("task-1", 7, onEvent);
  expect(new Headers(fetcher.mock.calls[0][1]?.headers).get("Last-Event-ID")).toBe("7");
  expect(onEvent).toHaveBeenCalledTimes(1);
  expect(onEvent).toHaveBeenCalledWith({ event_id: 8, kind: "agent_tool", data: { message: "页面已读取" } });
});
