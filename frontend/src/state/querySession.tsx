import { createContext, useContext, useRef, useState, type ReactNode } from "react";
import { useApi } from "../api/context";
import type { PostgreSqlVersion, QueryResponse } from "../api/types";
import { cacheEvidence } from "./evidenceCache";

interface QuerySession {
  question: string;
  setQuestion: (question: string) => void;
  version: PostgreSqlVersion | "";
  setVersion: (version: PostgreSqlVersion | "") => void;
  result: QueryResponse | null;
  error: string;
  loading: boolean;
  copied: boolean;
  submit: () => Promise<void>;
  cancel: () => void;
  copy: () => Promise<void>;
}

const QuerySessionContext = createContext<QuerySession | null>(null);

export function QuerySessionProvider({ children }: { children: ReactNode }) {
  const api = useApi();
  const [question, setQuestion] = useState("");
  const [version, setVersion] = useState<PostgreSqlVersion | "">("");
  const [result, setResult] = useState<QueryResponse | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [copied, setCopied] = useState(false);
  const pending = useRef<AbortController | null>(null);

  async function submit() {
    if (!question.trim()) { setError("请输入问题"); return; }
    pending.current?.abort();
    const controller = new AbortController();
    pending.current = controller;
    setError("");
    setResult(null);
    setCopied(false);
    setLoading(true);
    try {
      const response = await api.query(
        { question: question.trim(), requested_version: version || null },
        { signal: controller.signal },
      );
      setResult(response);
      const cited = new Set(response.citations.map((item) => item.evidence_id));
      response.evidence.forEach((item) => cacheEvidence(item, cited.has(item.evidence_id)));
    } catch (caught) {
      if (!(caught instanceof DOMException && caught.name === "AbortError")) {
        setError(caught instanceof Error ? caught.message : "未知错误");
      }
    } finally {
      if (pending.current === controller) {
        pending.current = null;
        setLoading(false);
      }
    }
  }

  async function copy() {
    if (!result?.answer) return;
    await navigator.clipboard.writeText(result.answer);
    setCopied(true);
  }

  const value: QuerySession = {
    question, setQuestion, version, setVersion, result, error, loading, copied,
    submit, cancel: () => pending.current?.abort(), copy,
  };
  return <QuerySessionContext.Provider value={value}>{children}</QuerySessionContext.Provider>;
}

export function useQuerySession() {
  const session = useContext(QuerySessionContext);
  if (!session) throw new Error("QuerySessionProvider is unavailable");
  return session;
}
