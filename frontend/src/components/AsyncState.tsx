import type { ReactNode } from "react";
export function LoadingState({ label = "正在读取数据…" }: { label?: string }) { return <div className="state-panel state-loading" role="status" aria-live="polite">{label}</div>; }
export function EmptyState({ title, children }: { title: string; children?: ReactNode }) { return <div className="state-panel"><strong>{title}</strong>{children && <p>{children}</p>}</div>; }
export function ErrorState({ message, retry }: { message: string; retry?: () => void }) { return <div className="state-panel state-error" role="alert"><strong>无法载入</strong><p>{message}</p>{retry && <button type="button" onClick={retry}>重试</button>}</div>; }
