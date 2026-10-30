import { expect, test } from "vitest";
import type { DiscoveryTask } from "../api/discovery";
import { suggestKnowledgeBaseId } from "./knowledgeBaseName";

test("uses the selected page title, strips site suffix and respects identifier limits", () => {
  const task = { product: "pandas", candidates: [
    { candidate_id: "one", title: "Extending pandas — pandas 3.1.0 documentation" },
    { candidate_id: "two", title: "User Guide — pandas documentation" },
  ] } as DiscoveryTask;
  expect(suggestKnowledgeBaseId(task, [])).toBe("extending-pandas");
  expect(suggestKnowledgeBaseId(task, ["two"])).toBe("pandas-user-guide");
  expect(suggestKnowledgeBaseId(task, ["one", "two"])).toBe("extending-pandas-collection");
  task.candidates[0].title = "a".repeat(100);
  expect(suggestKnowledgeBaseId(task, [])).toMatch(/^[a-z0-9][a-z0-9_-]{0,63}$/);
  task.candidates[0].title = "配置文档";
  expect(suggestKnowledgeBaseId(task, [])).toBe("pandas-docs");
});
