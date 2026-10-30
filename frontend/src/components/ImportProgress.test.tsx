import { render, screen, within } from "@testing-library/react";
import { expect, test } from "vitest";
import { ImportProgress } from "./ImportProgress";
import type { ImportJob } from "../api/discovery";
const job: ImportJob = { job_id: "job", task_id: "task", knowledge_base_id: "pandas", status: "running", result: null, failure_type: null };

test("shows the current stage and actual embedding counts instead of an overall estimate", () => {
  render(<ImportProgress job={{ ...job, progress: { phase: "embedding", completed: 320, total: 2251, unit: "chunks" } }} />);
  const panel = screen.getByRole("region", { name: "导入进度" });
  expect(within(panel).getByText("正在生成向量")).toBeVisible();
  expect(within(panel).getByText("320 / 2,251 个分块")).toBeVisible();
  expect(within(panel).getByRole("progressbar", { name: "当前阶段进度" })).toHaveAttribute("value", "320");
  expect(within(panel).getByText(/当前阶段 14%/)).toBeVisible();
});

test("collection counts are not misrepresented as a known final percentage", () => {
  render(<ImportProgress job={{ ...job, progress: { phase: "collecting", completed_pages: 20, discovered_pages: 30 } }} />);
  expect(screen.getByText("已读取 20 页 · 已发现 30 页")).toBeVisible();
  expect(screen.getByRole("progressbar")).not.toHaveAttribute("value");
});

test("completion and failure retain clear final states", () => {
  const { rerender } = render(<ImportProgress job={{ ...job, status: "succeeded", result: { document_count: 30, chunk_count: 2251 } }} />);
  expect(screen.getByText("导入完成，可以开始提问")).toBeVisible();
  expect(screen.getByText("30 页文档 · 2,251 个分块")).toBeVisible();
  rerender(<ImportProgress job={{ ...job, status: "failed", progress: { phase: "chunking", completed: 3, total: 30, unit: "pages" } }} />);
  expect(screen.getByText("导入失败 · 解析与切块")).toBeVisible();
});
