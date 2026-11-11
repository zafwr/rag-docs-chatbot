def test_health(client):
    assert client.get("/api/health").json() == {"status": "ok", "provider": "fake"}


def test_upload_list_chat_delete_flow(client, sample_pdf):
    upload = client.post("/api/documents", files={"file": ("handbook.pdf", sample_pdf, "application/pdf")})
    assert upload.status_code == 201
    doc = upload.json()
    assert doc["pages"] == 3 and doc["chunks"] > 0

    assert [d["id"] for d in client.get("/api/documents").json()] == [doc["id"]]

    chat = client.post("/api/chat", json={"question": "What is the home office budget?"})
    assert chat.status_code == 200
    body = chat.json()
    assert "500 euros" in body["answer"]
    assert body["sources"][0]["page"] == 3

    assert client.delete(f"/api/documents/{doc['id']}").status_code == 204
    assert client.get("/api/documents").json() == []


def test_rejects_unsupported_file(client):
    response = client.post("/api/documents", files={"file": ("photo.jpg", b"\xff\xd8", "image/jpeg")})
    assert response.status_code == 400


def test_rejects_empty_question(client):
    assert client.post("/api/chat", json={"question": ""}).status_code == 422


def test_delete_unknown_document_returns_404(client):
    assert client.delete("/api/documents/nope").status_code == 404
