export function StatusBadge({ tone, children }: { tone: "good" | "warn" | "bad" | "neutral"; children: string }) { return <span className={`status-badge status-${tone}`}>{children}</span>; }
