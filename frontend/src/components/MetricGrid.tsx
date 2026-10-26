export function MetricGrid({ metrics }: { metrics: Record<string, unknown> }) {
  const entries = Object.entries(metrics).filter(([, value]) => typeof value === "number");
  if (!entries.length) return <p className="muted">API 未返回数值指标。</p>;
  return <div className="metric-grid">{entries.map(([label, value]) => <div className="metric-card" key={label}><span>{label}</span><strong>{(value as number).toLocaleString(undefined, { maximumFractionDigits: 4 })}</strong></div>)}</div>;
}
