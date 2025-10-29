# AI Agent for Commerce Website

This project implements an AI-powered shopping assistant that can:
- Engage in **general conversation** with users
- Provide **text-based product recommendations** from a predefined catalog
- Perform **image-based product search**, returning matching items from the catalog

The goal is to deliver a single agent that can handle all 3 scenarios through a unified conversational interface.

---

## 🚀 Deployment

The agent is deployed on **Render** and can be accessed here:

👉 [Live Demo](https://ai-agent-for-commerce-website-1.onrender.com/)

👉 For full API details (requests, responses, examples), please see **[API.md](API.md)**.

---

## 🧩 Design Choices

### 1. Backend Model (OpenAI)
- **Choice**: We use the OpenAI GPT-4o model as the backend LLM.  
- **Reasoning**:
  - Provides strong natural language understanding for both casual conversation and structured queries on catalog. 
  - Supports **multi-modal inputs** (text + image), enabling image-based product search.  
  - Reliable API with good latency for real-time chat interfaces.  
  - The agent is designed to have engaging conversations with the customer by asking cross sell questions so the user is happy to stay and explore more.

- **Alternatives**:
  - Could use open-source models like LLaMA or Mistral, but these require hosting and fine-tuning infrastructure. However, open-source models are better in terms of privacy. It is an option worth considering.
  - I chose OpenAI for ease of deployment and speed for this exercise.

---

### 2. Product Catalog
- **Definition**: The catalog is a JSON file (`backend/data/catalog.json`) containing product entries with attributes:
  - `id`, `name`, `price`, `category`, and `image`.  
- **Choice**: Simple JSON format makes it easy to update and parse.  
- **Alternatives**:
  - A database (e.g., PostgreSQL, MongoDB) would be more scalable.  
  - For this exercise, JSON is sufficient and lightweight.

---

### 3. Product Search & Recommendation
- **Current Approach**: **Prompt engineering** – the LLM is instructed to generate recommendations by reasoning over the provided catalog context.  
- **Advantages**:
  - No external retrieval infra needed.  
  - Fast prototyping and flexible responses.  
- **Limitations**:
  - The LLM may occasionally hallucinate if not grounded properly.  
- **Alternatives**:
  - **RAG (Retrieval-Augmented Generation)**: Use embeddings + vector search to fetch candidate products, then let the LLM refine results.  
  - Would improve accuracy for larger catalogs.  
- **Decision**: Chose prompt engineering for simplicity, but I acknowledge that RAG would be a natural next step.

---

### 4. Frontend Design
- **Choice**: A lightweight HTML + JavaScript frontend:
  - Supports message bubbles for chat.  
  - Renders product recommendations as **cards** with images, price, and brand for better visualization.
  - Allows image upload with preview chip UI.  
- **Alternatives**:
  - Could use a frontend framework like React for scalability.  
  - For this exercise, plain HTML/JS is enough for clarity and deployment.

---

### 5. Session Memory
- **Choice**: Session context is stored in memory using Python `deque` per session ID. The agent has a memory of 50 messages.
- **Reasoning**: Keeps conversations coherent within a session.  
- **Alternatives**:
  - Persistent storage with database would be required for long-lived sessions or scaling.  
- **Decision**: In-memory storage is fine for a demo deployment.

---
