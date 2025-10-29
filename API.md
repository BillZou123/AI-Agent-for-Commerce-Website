# Agent API Documentation

This document describes the API endpoints for the AI-Powered Commerce Agent.  
The backend is a Flask app providing chat, image upload, and catalog access.

---


## Endpoints

| Method | Path                  | Description                                      |
|--------|-----------------------|--------------------------------------------------|
| GET    | `/`                   | Serve the frontend `index.html`                  |
| POST   | `/chat`               | Chat with the agent (text and/or image)          |
| POST   | `/upload`             | Upload an image, returns a public URL            |
| GET    | `/uploads/{filename}` | Serve an uploaded image                          |
| GET    | `/catalog`            | Return the product catalog JSON           |

---

## POST `/chat`

Chat with the AI agent. Accepts text and optional image.

### Request
```
curl -X POST https://ai-agent-for-commerce-website-1.onrender.com/chat \
  -H "Content-Type: application/json" \
  -d '{
        "message": "Show me laptops",
        "session_id": "12345-abcdef",
        "image_url": "/uploads/photo.jpg"

      }'

•	message (string, optional): User’s text message
•	session_id (string, optional): ID for conversation history. If missing, server generates one.
•	image_url (string, optional): Relative URL from /upload or an absolute URL.
```

### Response
{
  "reply": "I recommend the Apple MacBook Pro 14 (M2).",
  "session_id": "12345-abcdef",
  "items": [
    {
      "id": "p1",
      "name": "Apple MacBook Pro 14 (M2)",
      "price": 1999,
      "category": "electronics",
      "image": "/catalog_images/mbp14.jpg"
    }
  ]
}


## POST `/Upload`

Upload an image to the server and receive a public URL (to pass as image_url to /chat).

### Request
```
curl -X POST https://ai-agent-for-commerce-website-1.onrender.com/upload \
  -F "image=@/path/to/photo.jpg"

•	image (file, required): Multipart form-data field name is image (jpeg/png/webp, etc.)

```

### Response
{
“ok”: true,
“url”: “/uploads/photo_1.jpg”
}


## GET `/uploads/{filename}`

Fetch an uploaded file by filename (the same path returned by /upload).

### Request
```
curl https://ai-agent-for-commerce-website-1.onrender.com/uploads/photo_1.jpg --output photo_1.jpg

•	filename (path param, required): The exact filename (or nested path) under /uploads/ as returned by /upload.

```

### Response
(binary file content — e.g., image bytes)



## GET `/catalog`

Return the predefined product catalog JSON that the agent can recommend from.

### Request
```
curl https://ai-agent-fosr-commerce-website-1.onrender.com/catalog


```

### Response
{
“products”: [
{
“id”: “p1”,
“name”: “Apple MacBook Pro 14 (M2)”,
“price”: 1999,
“category”: “electronics”,
“image”: “/catalog_images/mbp14.jpg”
}
]
}


## GET `/`

Serve the frontend UI (index.html).

### Request
```
curl https://ai-agent-for-commerce-website-1.onrender.com/

```

### Response
(HTML document — the chat web app)
