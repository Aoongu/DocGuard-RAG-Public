import { createContext, useContext, type ReactNode } from "react";
import { apiClient, type ApiClient } from "./client";

const ApiContext = createContext<ApiClient>(apiClient);
export function ApiProvider({ client, children }: { client: ApiClient; children: ReactNode }) { return <ApiContext.Provider value={client}>{children}</ApiContext.Provider>; }
export function useApi() { return useContext(ApiContext); }
