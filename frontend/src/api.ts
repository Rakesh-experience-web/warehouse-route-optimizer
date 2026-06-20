import { MapLayoutPayload, OptimizationRequest, OptimizationResponse, StoredMap } from "./types";

function resolveBaseUrl(): string {
  const envBase = import.meta.env.VITE_API_BASE_URL as string | undefined;
  if (envBase && envBase.trim().length > 0) {
    return envBase.replace(/\/+$/, "");
  }
  const host = window.location.hostname || "127.0.0.1";
  return `http://${host}:8000/api/v1`;
}

const BASE_URL = resolveBaseUrl();

export async function optimize(payload: OptimizationRequest): Promise<OptimizationResponse> {
  try {
    const res = await fetch(`${BASE_URL}/optimize`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    if (!res.ok) {
      const body = await res.text();
      throw new Error(`Optimization failed: ${body}`);
    }
    return (await res.json()) as OptimizationResponse;
  } catch (err) {
    if (err instanceof TypeError) {
      throw new Error(
        `Cannot reach backend at ${BASE_URL}. Ensure FastAPI is running and CORS allows your frontend origin.`
      );
    }
    throw err;
  }
}

export async function saveMap(payload: MapLayoutPayload): Promise<StoredMap> {
  const res = await fetch(`${BASE_URL}/maps`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`Map save failed: ${body}`);
  }
  return (await res.json()) as StoredMap;
}

export async function listMaps(): Promise<StoredMap[]> {
  const res = await fetch(`${BASE_URL}/maps`);
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`Map list failed: ${body}`);
  }
  const body = (await res.json()) as { maps: StoredMap[] };
  return body.maps;
}

export async function getMap(mapId: string): Promise<StoredMap> {
  const res = await fetch(`${BASE_URL}/maps/${mapId}`);
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`Map load failed: ${body}`);
  }
  return (await res.json()) as StoredMap;
}
