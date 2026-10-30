import type { DiscoveryTask } from "../api/discovery";

export function suggestKnowledgeBaseId(task: DiscoveryTask, selected: string[]): string {
  const candidates = task.candidates.filter(c => selected.includes(c.candidate_id));
  const page = candidates[0] ?? task.candidates[0];
  const slug = (text: string) => text.normalize("NFKD").replace(/[\u0300-\u036f]/g, "").toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "");
  const title = page?.title.split(/\s+[—–|]\s+/)[0] ?? "";
  const product = slug(task.product);
  let name = slug(title);
  if (name && product && !name.includes(product)) name = `${product}-${name}`;
  if (!name && product) name = `${product}-docs`;
  if (!name) name = "technical-docs";
  if (candidates.length > 1) name += "-collection";
  return name.slice(0, 64).replace(/-+$/g, "");
}
