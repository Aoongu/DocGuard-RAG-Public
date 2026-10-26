import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { expect, test } from "vitest";
import { ApiProvider } from "../api/context";
import { AppRoutes } from "../App";
import { fakeApi } from "./fakeApi";

test("provides landmarks, named navigation, labels, and live service status", async () => {
  const api = fakeApi({ getHealth: async () => ({ request_id: "h", status: "ok", dependencies: [] }) });
  render(<MemoryRouter initialEntries={["/query"]}><ApiProvider client={api}><AppRoutes /></ApiProvider></MemoryRouter>);
  expect(screen.getByRole("navigation")).toBeInTheDocument();
  expect(screen.getByRole("main")).toBeInTheDocument();
  expect(screen.getByLabelText("问题")).toBeInTheDocument();
  expect(screen.getByLabelText("目标版本")).toBeInTheDocument();
  expect(await screen.findByText("服务正常")).toBeInTheDocument();
});
