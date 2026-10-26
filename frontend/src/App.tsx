import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { AppShell } from "./components/AppShell";
import { EvidencePage } from "./pages/EvidencePage";
import { EvaluationsPage } from "./pages/EvaluationsPage";
import { KnowledgePage } from "./pages/KnowledgePage";
import { QueryPage } from "./pages/QueryPage";
import { QuerySessionProvider } from "./state/querySession";

export function AppRoutes() { return <QuerySessionProvider><AppShell><Routes><Route path="/knowledge" element={<KnowledgePage />} /><Route path="/query" element={<QueryPage />} /><Route path="/evidence" element={<EvidencePage />} /><Route path="/evidence/:evidenceId" element={<EvidencePage />} /><Route path="/evaluations" element={<EvaluationsPage />} /><Route path="*" element={<Navigate replace to="/knowledge" />} /></Routes></AppShell></QuerySessionProvider>; }
export function App() { return <BrowserRouter><AppRoutes /></BrowserRouter>; }
