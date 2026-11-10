// Thin wrapper around the ScribeScan JSON API.

export class ApiError extends Error {
  constructor(status, body) {
    super((body && body.detail && typeof body.detail === "string") ? body.detail : `Request failed (${status})`);
    this.status = status;
    this.body = body;
  }
}

async function request(method, url, body, { keepalive = false } = {}) {
  let res;
  try {
    res = await fetch(url, {
      method,
      headers: body !== undefined ? { "Content-Type": "application/json" } : {},
      body: body !== undefined ? JSON.stringify(body) : undefined,
      credentials: "same-origin",
      keepalive,
    });
  } catch (err) {
    throw new ApiError(0, { detail: "Can't reach the server — check your connection." });
  }
  if (res.status === 401) {
    // Access-token cookie missing/expired: reload to show the server's lock page.
    window.location.reload();
  }
  const isJson = (res.headers.get("content-type") || "").includes("application/json");
  const data = isJson ? await res.json() : null;
  if (!res.ok) throw new ApiError(res.status, data);
  return data;
}

export const api = {
  health: () => request("GET", "/api/health"),
  listDocuments: () => request("GET", "/api/documents"),
  createDocument: (title) => request("POST", "/api/documents", title ? { title } : {}),
  getDocument: (id) => request("GET", `/api/documents/${id}`),
  updateDocument: (id, payload, opts) => request("PUT", `/api/documents/${id}`, payload, opts),
  deleteDocument: (id) => request("DELETE", `/api/documents/${id}`),
  scan: (text, documentId) => request("POST", "/api/scan", { text, document_id: documentId ?? null }),
};
