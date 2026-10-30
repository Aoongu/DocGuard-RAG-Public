import { render, screen } from "@testing-library/react";
import { expect, test } from "vitest";
import { CitedAnswer } from "./CitedAnswer";

test("unregistered references remain visible and never link to a different evidence", () => {
  render(<CitedAnswer text="Known [E1], unregistered [E2]." labels={new Set(["E01"])} />);
  expect(screen.getByRole("link", { name: "E01" })).toHaveAttribute("href", "#citation-E01");
  expect(screen.queryByRole("link", { name: "E02" })).not.toBeInTheDocument();
  expect(screen.getByText("E2（未登记引用）")).toBeInTheDocument();
});
