export function citationLabel(value: string): string | undefined {
  const match = /^E(\d{1,3})$/.exec(value);
  const number = match && Number(match[1]);
  return number && number >= 1 && number <= 10 ? "E" + String(number).padStart(2, "0") : undefined;
}

export function CitedAnswer({ text, labels }: { text: string; labels: Set<string> }) {
  return <p id="answer-text" tabIndex={-1}>{text.split(/(\bE\d{1,3}\b)/g).map((part, index) => {
    if (!/^E\d{1,3}$/.test(part)) return part;
    const label = citationLabel(part);
    if (!label || !labels.has(label)) return <span key={index} className="citation-unresolved" title="此编号未出现在后端确认的引用列表中">{part}（未登记引用）</span>;
    return <a key={index} className="inline-citation" href={"#citation-" + label} onClick={event => {
      event.preventDefault();
      const target = document.getElementById("citation-" + label);
      target?.scrollIntoView?.({ block: "center" });
      target?.focus({ preventScroll: true });
    }}>{label}</a>;
  })}</p>;
}
