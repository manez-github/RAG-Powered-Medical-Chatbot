# 🏥 RAG-Powered Medical Chatbot

A **production-deployed** Retrieval-Augmented Generation (**RAG**) chatbot, live on **AWS EC2**, that answers medical questions through a full-stack pipeline — from a **HTML / CSS / JS / jQuery** frontend, through **Python Flask APIs**, into a **RAG** core powered by **HuggingFace Embeddings** + **Pinecone** vector search + **Groq (LLaMA 3.3)** — containerized with **Docker** and shipped automatically via a **CI/CD** pipeline into **AWS ECR → EC2**.

> **Stack:** `HTML` · `CSS` · `JavaScript` · `jQuery` · `AJAX` · `Bootstrap` · `Python` · `Flask API` · `Pinecone` · `HuggingFace Embeddings` · `Groq (LLaMA 3.3)` · ⭐ `RAG` · `Docker` · `CI/CD` · `AWS EC2` · `AWS ECR`

---

### 🔍 How This Was Built — The Full Story

The frontend is a conversational chat interface built with **HTML, CSS, JavaScript, jQuery, AJAX, and Bootstrap**. When a user types a question and hits send, **AJAX** fires an asynchronous call to a **Flask API** endpoint — no page reload, just a smooth chat experience. On the backend, Flask receives the question via the request object and passes it straight into the RAG chain.

The knowledge base powering the chatbot is the **Gale Encyclopedia of Medicine** — a comprehensive reference covering virtually every disease, symptom, treatment, and medical concept. This book was loaded as a PDF, parsed into individual documents, and then split into small chunks. Each chunk was passed through a **HuggingFace sentence-transformers embedding model** which converted the text into numerical vectors that capture its semantic meaning. All those vectors were then stored in a **Pinecone index**, turning the encyclopedia into a searchable vector database.

The heart of the project is the **RAG chain** — built with LangChain by wiring together three components: a **retriever** (Pinecone similarity search that fetches the top-3 most relevant chunks for any question), an **LLM** (ChatGroq running LLaMA 3.3 70B), and a **prompt template** that instructs the model to answer concisely using only the retrieved context. The chain is fully runnable — you pass in a question, you get a grounded answer. If the retrieved context doesn't contain the answer, the model says so rather than guessing.

For deployment, an **AWS ECR** repository was created to store Docker images, and an **AWS EC2** instance was set up to serve the application. AWS access keys were generated, and all sensitive credentials — including Pinecone and Groq API keys — were stored as **GitHub Secrets**. The EC2 instance was registered as a **self-hosted GitHub Actions runner**, making it the target machine for deployments. A **Dockerfile** packages the Flask app into a portable container, and a **`cicd.yml`** GitHub Actions workflow automates the entire process with two jobs: the **CI job** (runs on a GitHub-hosted runner) builds the Docker image and pushes it to ECR, and the **CD job** (runs on the EC2 self-hosted runner) pulls that image from ECR and runs it on the server. Every `git push` to `main` triggers the pipeline automatically — and the chatbot is live and accessible via the EC2 public IP.

---

## 🚀 Live Demo

> Ask the chatbot anything medical — symptoms, treatments, drugs, anatomy — and it will retrieve the most relevant passages from its knowledge base and generate a concise, grounded answer.

---

## 🧠 How It Works

```
User Question
     │
     ▼
 Flask App (app.py)
     │
     ▼
 HuggingFace Embeddings          ◄─── sentence-transformers/all-MiniLM-L6-v2
 (Convert question to vector)
     │
     ▼
 Pinecone Vector Store           ◄─── Pre-indexed medical PDF chunks
 (Similarity search, top-k=3)
     │
     ▼
 Retrieved Context (chunks)
     │
     ▼
 Groq LLM (LLaMA 3.3 70B)       ◄─── Generates answer using context
     │
     ▼
 Answer → User
```

The chatbot **never hallucinates facts it doesn't have** — if the retrieved context doesn't contain the answer, it says so.

---


## 📁 Directory Structure

```
RAG-Powered-Medical-Chatbot/
│
├── .github/
│   └── workflows/
│       └── cicd.yml              # GitHub Actions CI/CD pipeline
│
├── data/                         
│   └── Medical_book.pdf          # Medical PDF: The Gale Encyclopedia of medicine
│
├── research/
│   └── trials.ipynb              # Jupyter notebook for RAG experimentation
│
├── src/
│   ├── helper.py                 # PDF loading, chunking, embedding utilities
│   └── prompt.py                 # System prompt for the medical assistant
│
├── static/
│   └── style.css                 # Chat UI styles
│
├── templates/
│   └── index.html                # Chat UI (HTML + Bootstrap + jQuery)
│
├── app.py                        # Flask web application (main entry point)
├── store_index.py                # One-time script to embed PDFs → Pinecone
├── Dockerfile                    # Containerization config
├── pyproject.toml                # Python dependencies
├── template.sh                   # Shell script to scaffold project structure
├── uv.lock                       # Lockfile for reproducible installs
├── .python-version               # Pinned Python version
├── .gitignore
└── README.md
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **LLM** | Groq API — `llama-3.3-70b-versatile` |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace) |
| **Vector Store** | Pinecone (Serverless, AWS `us-east-1`) |
| **RAG Framework** | LangChain |
| **Web Framework** | Flask |
| **Frontend** | HTML + CSS + Bootstrap 4 + jQuery |
| **Containerization** | Docker |
| **Cloud** | AWS EC2 + AWS ECR |
| **CI/CD** | GitHub Actions |

---

## ⚙️ Setup & Installation

### Prerequisites

- Python >= 3.10
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- A [Pinecone](https://www.pinecone.io/) account and API key
- A [Groq](https://console.groq.com/) account and API key

### 1. Clone the repository

```bash
git clone https://github.com/manez-github/RAG-Powered-Medical-Chatbot.git
cd RAG-Powered-Medical-Chatbot
```

### 2. Install dependencies

```bash
# Using uv (recommended)
uv sync 
```

### 3. Configure environment variables

Create a `.env` file in the root directory:

```env
PINECONE_API_KEY=your_pinecone_api_key_here
GROQ_API_KEY=your_groq_api_key_here
AWS_ACCESS_KEY_ID=your_aws_key         # Only needed for deployment
AWS_SECRET_ACCESS_KEY=your_aws_secret  # Only needed for deployment
AWS_DEFAULT_REGION=your_aws_region     # Only needed for deployment
```

### 4. Add your medical PDFs

Place your PDF files in the `data/` directory. 

### 5. Index the PDFs into Pinecone

> **Run this only once** (or whenever you add new PDFs to the knowledge base).

```bash
python store_index.py
```

This will:
- Load and parse all PDFs from `data/`
- Split them into 500-token chunks (20-token overlap)
- Generate embeddings using the MiniLM model
- Create a `medical-chatbot` index in Pinecone and upload all vectors

### 6. Run the application

```bash
python app.py
```

The app will be available at `http://localhost:8000`.

---

## 🔄 CI/CD Pipeline

The project uses GitHub Actions for automated build and deployment to AWS EC2.

**Continuous Integration** (runs on GitHub-hosted runner):
1. Checks out the code
2. Authenticates with AWS
3. Builds the Docker image
4. Pushes it to Amazon ECR

**Continuous Deployment** (runs on self-hosted EC2 runner):
1. Pulls the latest image from ECR
2. Stops and removes any existing containers
3. Runs the new container, injecting all secrets as environment variables

### Required GitHub Secrets

| Secret | Description |
|---|---|
| `AWS_ACCESS_KEY_ID` | AWS IAM access key |
| `AWS_SECRET_ACCESS_KEY` | AWS IAM secret key |
| `AWS_DEFAULT_REGION` | AWS region (e.g., `us-east-1`) |
| `ECR_REPO` | ECR repository name |
| `PINECONE_API_KEY` | Pinecone API key |
| `GROQ_API_KEY` | Groq API key |

---

## 🔬 Research & Experimentation

The `research/trials.ipynb` notebook contains exploratory work used to prototype and validate the RAG pipeline before productionizing it — including chunk size tuning, embedding evaluation, and retrieval quality checks.

---

## 📦 Key Dependencies

```toml
langchain==0.3.26
langchain-pinecone==0.2.8
langchain-groq>=0.3.8
langchain-community==0.3.26
sentence-transformers==4.1.0
flask==3.1.1
pypdf==5.6.1
python-dotenv==1.1.0
gunicorn>=25.1.0
```

---

## 🗂️ Core Module Reference

### `src/helper.py`

| Function | Description |
|---|---|
| `load_pdf_files(path)` | Recursively loads all PDFs from a directory using `PyPDFLoader` |
| `filter_to_minimal_docs(docs)` | Strips metadata down to `source` only for cleaner storage |
| `download_embeddings()` | Downloads and returns the `all-MiniLM-L6-v2` HuggingFace model |
| `create_chunks(docs)` | Splits documents into 500-token chunks with 20-token overlap |

### `src/prompt.py`

Defines the system prompt for the medical assistant. The LLM is instructed to answer concisely using only the retrieved context, and to admit when it doesn't know the answer.

### `store_index.py`

One-time ingestion script. Creates the Pinecone index if it doesn't exist and populates it with vectorized document chunks.

### `app.py`

Flask application. On startup, it connects to the existing Pinecone index and builds the RAG chain. Serves the chat UI at `/` and handles message POST requests at `/get`.

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change.

---

## 📄 License

This project is open-source. See the repository for license details.