import { createContext, useContext, useEffect, useRef, useState, type ReactNode } from "react";
import { useApi } from "../api/context";
import type { PostgreSqlVersion, QueryResponse } from "../api/types";
import { cacheEvidence, cacheKnowledgeEvidence } from "./evidenceCache";
import type { KnowledgeAnswer } from "../api/discovery";

interface QuerySession {
  generateKnowledgeAnswer: boolean;
  setGenerateKnowledgeAnswer: (value: boolean) => void;
  knowledgeBaseId: string;
  setKnowledgeBaseId: (id: string) => void;
  knowledgeAnswer: KnowledgeAnswer | null;
  answeredKnowledgeBaseId: string;
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
  const [knowledgeBaseId, setKnowledgeBaseId] = useState("");
  const [generateKnowledgeAnswer, setGenerateKnowledgeAnswer] = useState(true);
  const [knowledgeAnswer, setKnowledgeAnswer] = useState<KnowledgeAnswer | null>(null);
  const [answeredKnowledgeBaseId, setAnsweredKnowledgeBaseId] = useState("");
  const [version, setVersion] = useState<PostgreSqlVersion | "">("");
  const [result, setResult] = useState<QueryResponse | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [copied, setCopied] = useState(false);
  const pending = useRef<AbortController | null>(null);
  useEffect(() => {
    const deleted = (event: Event) => {
      const id = (event as CustomEvent<string>).detail;
      if (id === knowledgeBaseId) {
        pending.current?.abort(); pending.current = null;
        setKnowledgeBaseId(""); setLoading(false); setError("");
      }
      if (id === answeredKnowledgeBaseId) { setKnowledgeAnswer(null); setAnsweredKnowledgeBaseId(""); }
    };
    window.addEventListener("docguard:knowledge-deleted", deleted);
    return () => window.removeEventListener("docguard:knowledge-deleted", deleted);
  }, [knowledgeBaseId, answeredKnowledgeBaseId]);

  async function submit() {
    if (!question.trim()) { setError("请输入问题"); return; }
    pending.current?.abort();
    const controller = new AbortController();
    pending.current = controller;
    setError("");
    setResult(null);
    setKnowledgeAnswer(null);
    setCopied(false);
    setLoading(true);
    try {
      if (knowledgeBaseId) {
        const response = await api.queryKnowledgeBase(knowledgeBaseId, question.trim(), { signal: controller.signal, generate: generateKnowledgeAnswer });
        if (pending.current === controller && !controller.signal.aborted) {
          setKnowledgeAnswer(response); setAnsweredKnowledgeBaseId(knowledgeBaseId);
          (response.evidence ?? []).forEach(item => cacheKnowledgeEvidence(knowledgeBaseId, item));
          response.citations.forEach(item => cacheKnowledgeEvidence(knowledgeBaseId, item));
        }
        return;
      }
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
    const answer = knowledgeBaseId ? knowledgeAnswer?.answer : result?.answer;
    if (!answer) return;
    await navigator.clipboard.writeText(answer);
    setCopied(true);
  }

  const value: QuerySession = {
    generateKnowledgeAnswer, setGenerateKnowledgeAnswer,
    knowledgeBaseId, setKnowledgeBaseId, knowledgeAnswer, answeredKnowledgeBaseId,
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
