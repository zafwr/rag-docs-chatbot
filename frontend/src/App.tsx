import { FormEvent, useEffect, useRef, useState } from "react";
import { api, ChatAnswer, DocumentInfo, Source } from "./api";

type Message =
  | { role: "user"; text: string }
  | { role: "assistant"; text: string; sources: Source[] }
  | { role: "error"; text: string };

const EXAMPLE_QUESTIONS = [
  "How many days of paid leave do I get?",
  "What is the home office budget?",
  "When are core working hours?",
];

export default function App() {
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [messages, setMessages] = useState<Message[]>([]);
  const [question, setQuestion] = useState("");
  const [uploading, setUploading] = useState(false);
  const [thinking, setThinking] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const chatEnd = useRef<HTMLDivElement>(null);

  useEffect(() => {
    api.listDocuments().then(setDocuments).catch(() => setDocuments([]));
  }, []);

  useEffect(() => {
    chatEnd.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, thinking]);

  async function handleUpload(files: FileList | null) {
    if (!files?.length) return;
    setUploading(true);
    setUploadError(null);
    try {
      for (const file of Array.from(files)) {
        const doc = await api.uploadDocument(file);
        setDocuments((current) => [...current, doc]);
      }
    } catch (error) {
      setUploadError((error as Error).message);
    } finally {
      setUploading(false);
      if (fileInput.current) fileInput.current.value = "";
    }
  }

  async function handleDelete(id: string) {
    await api.deleteDocument(id);
    setDocuments((current) => current.filter((d) => d.id !== id));
  }

  async function ask(text: string) {
    const trimmed = text.trim();
    if (!trimmed || thinking) return;
    setMessages((current) => [...current, { role: "user", text: trimmed }]);
    setQuestion("");
    setThinking(true);
    try {
      const result: ChatAnswer = await api.ask(trimmed);
      setMessages((current) => [...current, { role: "assistant", text: result.answer, sources: result.sources }]);
    } catch (error) {
      setMessages((current) => [...current, { role: "error", text: (error as Error).message }]);
    } finally {
      setThinking(false);
    }
  }

  function handleSubmit(event: FormEvent) {
    event.preventDefault();
    ask(question);
  }

  return (
    <div className="layout">
      <aside className="sidebar">
        <h1>Docs Chatbot</h1>
        <p className="muted">Upload documents, then ask questions. Every answer cites the page it came from.</p>

        <label className={`dropzone ${uploading ? "busy" : ""}`}>
          <input
            ref={fileInput}
            type="file"
            accept=".pdf,.txt,.md"
            multiple
            disabled={uploading}
            onChange={(e) => handleUpload(e.target.files)}
          />
          {uploading ? "Processing…" : "+ Upload PDF, TXT or MD"}
        </label>
        {uploadError && <p className="error-text">{uploadError}</p>}

        <h2>Documents</h2>
        {documents.length === 0 ? (
          <p className="muted">No documents yet.</p>
        ) : (
          <ul className="doc-list">
            {documents.map((doc) => (
              <li key={doc.id}>
                <div>
                  <strong title={doc.name}>{doc.name}</strong>
                  <span className="muted">
                    {doc.pages} pages · {doc.chunks} chunks
                  </span>
                </div>
                <button className="icon" aria-label={`Remove ${doc.name}`} onClick={() => handleDelete(doc.id)}>
                  ×
                </button>
              </li>
            ))}
          </ul>
        )}
      </aside>

      <main className="chat">
        <div className="messages">
          {messages.length === 0 && (
            <div className="empty">
              <p>Ask anything about your documents.</p>
              {documents.length > 0 && (
                <div className="examples">
                  {EXAMPLE_QUESTIONS.map((q) => (
                    <button key={q} onClick={() => ask(q)}>
                      {q}
                    </button>
                  ))}
                </div>
              )}
            </div>
          )}

          {messages.map((message, index) => (
            <div key={index} className={`message ${message.role}`}>
              <p>{message.text}</p>
              {message.role === "assistant" && message.sources.length > 0 && (
                <details className="sources">
                  <summary>
                    {message.sources.length} {message.sources.length === 1 ? "source" : "sources"}
                  </summary>
                  {message.sources.map((s) => (
                    <blockquote key={s.number}>
                      <cite>
                        [{s.number}] {s.document_name}, page {s.page}
                      </cite>
                      {s.text.length > 320 ? `${s.text.slice(0, 320)}…` : s.text}
                    </blockquote>
                  ))}
                </details>
              )}
            </div>
          ))}
          {thinking && <div className="message assistant thinking">Searching your documents…</div>}
          <div ref={chatEnd} />
        </div>

        <form className="composer" onSubmit={handleSubmit}>
          <input
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder={documents.length ? "Ask a question…" : "Upload a document to start"}
            disabled={thinking}
          />
          <button type="submit" disabled={!question.trim() || thinking}>
            Ask
          </button>
        </form>
      </main>
    </div>
  );
}
