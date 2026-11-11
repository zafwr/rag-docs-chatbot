export type DocumentInfo = {
  id: string;
  name: string;
  pages: number;
  chunks: number;
};

export type Source = {
  number: number;
  document_name: string;
  page: number;
  text: string;
  score: number;
};

export type ChatAnswer = {
  answer: string;
  sources: Source[];
};

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`/api${path}`, init);
  if (!response.ok) {
    let message = `Request failed (${response.status})`;
    try {
      const body = await response.json();
      if (typeof body.detail === "string") message = body.detail;
    } catch {
      // Response had no JSON body; keep the generic message.
    }
    throw new Error(message);
  }
  return response.status === 204 ? (undefined as T) : response.json();
}

export const api = {
  listDocuments: () => request<DocumentInfo[]>("/documents"),

  uploadDocument: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<DocumentInfo>("/documents", { method: "POST", body: form });
  },

  deleteDocument: (id: string) => request<void>(`/documents/${id}`, { method: "DELETE" }),

  ask: (question: string) =>
    request<ChatAnswer>("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    }),
};
