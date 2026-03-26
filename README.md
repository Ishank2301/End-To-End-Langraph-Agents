

# 💊 MediBot — Medical Information Chatbot

> A production-grade RAG-based medicinal chatbot powered by **LangGraph**, **LangChain**, **FAISS**, and **MLflow** with full CI/CD automation.

⚠️ **Disclaimer:** MediBot is for educational and informational purposes only. It does NOT provide medical advice. Always consult a qualified healthcare professional.

---

## 🏗️ Architecture

```
User Query
    ↓
Streamlit UI (app.py)
    ↓
LangGraph Conversation Flow
    ├── Node 1: Intent Classifier
    │     └── emergency | medical_query | greeting | out_of_scope
    ├── Node 2: Document Retriever (FAISS) ← skipped for non-medical
    ├── Node 3: Response Generator (LangChain RAG Chain)
    └── Node 4: Safety Checker + Disclaimer Injector
    ↓
MLflow Tracking (latency, intent, confidence, sources)
    ↓
Final Grounded Response with Citations
```

---

## 🚀 Features

| Feature | Description |
|---|---|
| 🧠 LangGraph Stateful Flow | 4-node graph: classify → retrieve → generate → safety check |
| 🔎 RAG with FAISS | Grounds every answer in your medical document knowledge base |
| 🔄 Multi-LLM Support | Switch between OpenAI GPT and Ollama (local) via `.env` |
| 📊 MLflow Tracking | Logs every interaction: latency, intent, confidence, sources |
| 🚨 Emergency Detection | Detects emergency keywords and redirects to emergency services |
| 🛡️ Safety Layer | Auto-injects disclaimers, validates all responses |
| 🤖 GitHub PR Bot | Auto labels, size checks, review checklists on every PR |
| 🔁 Dependabot | Weekly automated dependency updates |
| 🐳 Docker Ready | One-command containerized deployment |

---

## ⚙️ Installation

### 1. Clone the repo
```bash
git clone https://github.com/your-username/medicinal-chatbot.git
cd medicinal-chatbot
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment
```bash
cp .env.example .env
# Edit .env with your settings
```

### 5. Add medical documents
```bash
# Place your PDFs and .txt files in the data/ folder
cp your_medical_docs/*.pdf data/
```

---

## ▶️ Running the App

### Streamlit UI
```bash
streamlit run app.py
```

### With MLflow Tracking
```bash
# Start MLflow server first
mlflow server --host 0.0.0.0 --port 5000

# Then run app
streamlit run app.py
```

### Docker
```bash
docker build -t medibot .
docker run -p 8501:8501 --env-file .env medibot
```

---

## 🧪 Running Tests
```bash
# All tests with coverage
pytest tests/ -v --cov=. --cov-report=term-missing

# Single test file
pytest tests/test_medibot.py -v
```

---

## 🔄 CI/CD Pipeline

Every push/PR triggers:

| Stage | Action |
|---|---|
| 🔍 Lint | Black, isort, Flake8 |
| 🧪 Test | pytest + coverage report |
| 🔐 Security | Bandit scan |
| 🐳 Docker | Build & push to Docker Hub (main branch only) |
| 🚀 Deploy | Auto-deploy on successful main merge |

### GitHub Secrets Required
Add these in **Settings → Secrets → Actions**:
- `DOCKER_USERNAME`
- `DOCKER_PASSWORD`
- `OPENAI_API_KEY` (if using OpenAI)

---

## 📊 MLflow Dashboard

View experiment runs at `http://localhost:5000` after starting the MLflow server.

**Tracked metrics per interaction:**
- `latency_ms` — response time
- `confidence_score` — intent confidence
- `num_retrieved_docs` — RAG effectiveness
- `response_length` — output verbosity

---

## 📁 Project Structure

```
medicinal-chatbot/
├── app.py                          # Streamlit UI entry point
├── chains/
│   └── rag_chain.py                # RAG chain (embeddings, retrieval, LLM)
├── graph/
│   └── chatbot_graph.py            # LangGraph 4-node conversation flow
├── mlflow_tracking/
│   └── tracker.py                  # MLflow interaction logging
├── utils/
│   ├── config.py                   # Centralized settings
│   └── logger.py                   # Structured logging
├── tests/
│   └── test_medibot.py             # Unit tests
├── data/                           # Add your medical PDFs here
├── embeddings/                     # Auto-generated FAISS index
├── .github/
│   ├── workflows/
│   │   ├── ci_cd.yml               # Main CI/CD pipeline
│   │   └── pr_bot.yml              # PR review bot
│   └── dependabot.yml              # Auto dependency updates
├── Dockerfile
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🔮 Roadmap
- [ ] Medicine interaction checker
- [ ] Multi-language support (Hindi, Spanish)
- [ ] FastAPI backend + React frontend
- [ ] Voice input support
- [ ] Reranker for better retrieval quality
- [ ] Evaluation pipeline with RAGAS

---

## 🤝 Contributing
1. Fork the repo
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit: `git commit -m "feat: your feature"`
4. Push and open a PR — the bot will auto-review!

---

## 📜 License
MIT License — see [LICENSE](LICENSE)

---

## 🙌 Acknowledgements
- LangChain & LangGraph teams
- MLflow open-source community
- Original project inspiration from End-To-End-Langraph-Agents
