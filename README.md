.

🧠 Medicinal Chatbot using LangGraph, LangChain & RAG

A Retrieval-Augmented Generation (RAG) based medicinal chatbot designed to answer medical and pharmaceutical queries using domain-specific knowledge sources.
This project leverages LangChain for LLM orchestration and LangGraph for building a structured, stateful conversational workflow.

⚠️ Disclaimer: This chatbot is for educational and informational purposes only and does not provide medical advice.

🚀 Features

💊 Answers medicine-related questions using retrieved documents

🔎 Uses RAG (Retrieval-Augmented Generation) for factual grounding

🧩 Modular conversational flow using LangGraph

🧠 Context-aware multi-turn conversations

📚 Custom medical knowledge base (PDFs / text / docs)

🛡️ Reduces hallucinations by grounding responses in sources

🏗️ Project Architecture
User Query
    ↓
LangGraph (Conversation Flow)
    ↓
Retriever (Vector Database)
    ↓
Relevant Medical Documents
    ↓
LLM (via LangChain)
    ↓
Final Answer (Grounded Response)

🧰 Tech Stack

Python

LangChain – LLM orchestration

LangGraph – Graph-based conversational flow

Vector Database (FAISS / Chroma)

Embedding Model (OpenAI / HuggingFace)

LLM (OpenAI / local LLM)

Document Loaders (PDF, Text, etc.)

📂 Project Structure
medicinal-chatbot/
│
├── data/                 # Medical documents (PDFs, text files)
├── embeddings/           # Vector store files
├── graph/                # LangGraph workflow
│   └── chatbot_graph.py
├── chains/               # LangChain RAG chains
├── utils/                # Helper functions
├── app.py                # Main application entry point
├── requirements.txt
└── README.md

⚙️ Installation
1️⃣ Clone the repository
git clone https://github.com/your-username/medicinal-chatbot.git
cd medicinal-chatbot

2️⃣ Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

3️⃣ Install dependencies
pip install -r requirements.txt

🔐 Environment Variables

Create a .env file and add:

OPENAI_API_KEY=your_api_key_here


(Modify if using a local or different LLM provider)

▶️ How to Run
python app.py


Once running, you can interact with the medicinal chatbot via terminal or UI (if integrated).

💡 Example Queries

What is Paracetamol used for?

Side effects of Ibuprofen

Can this medicine be taken during pregnancy?

Difference between antibiotics and antivirals

🧠 Why LangGraph?

LangGraph enables:

Explicit control over conversation states

Deterministic and debuggable workflows

Easy extension (memory, tools, validators)

Cleaner architecture than linear chains

🛡️ Safety & Limitations

Not a replacement for professional medical advice

Responses depend on provided documents

May not cover rare or emergency conditions

🔮 Future Improvements

✅ Source citations in responses

✅ Medicine interaction checker

✅ Doctor-style follow-up questions

✅ Web UI (Streamlit / FastAPI)

✅ Multi-language support

🤝 Contributing

Contributions are welcome!
Feel free to fork the repo, open issues, or submit PRs.

📜 License

This project is licensed under the MIT License.

🙌 Acknowledgements

LangChain & LangGraph communities

Open-source LLM ecosystem

Medical data sources used for learning purposes
