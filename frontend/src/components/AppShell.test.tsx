import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { expect, test } from "vitest";

import { AppRoutes } from "../App";

test("exposes all four primary destinations and a skip link", () => {
  render(
    <MemoryRouter initialEntries={["/knowledge"]}>
      <AppRoutes />
    </MemoryRouter>,
  );

  expect(screen.getAllByRole("link", { name: /知识库|技术问答|证据检查|评测看板/ })).toHaveLength(4);
  expect(screen.getByRole("link", { name: "跳到主要内容" })).toHaveAttribute("href", "#main-content");
  expect(screen.getByRole("main")).toHaveAttribute("id", "main-content");
});
