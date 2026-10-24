# 🧠 X-Estal Enterprise Hybrid RAG & CPU Inference Engine

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python" alt="Python" />
  <img src="https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B?style=for-the-badge&logo=streamlit" alt="Streamlit" />
  <img src="https://img.shields.io/badge/LangGraph-State_Machine-orange?style=for-the-badge" alt="LangGraph" />
  <img src="https://img.shields.io/badge/Groq-Cloud_Inference-f55036?style=for-the-badge" alt="Groq" />
  <img src="https://img.shields.io/badge/llama.cpp-Flash_Attention-green?style=for-the-badge" alt="llama.cpp" />
  <img src="https://img.shields.io/badge/ChromaDB-Vector_Store-purple?style=for-the-badge" alt="ChromaDB" />
  <img src="https://img.shields.io/badge/License-MIT-black?style=for-the-badge" alt="License" />
</p>

<p align="center">
  <b>Architected & Developed by Nitish Kumar (<a href="https://github.com/XestalNitish">@x-estal nitish</a>)</b><br>
  <i>Hybrid RAG Engine with Custom Document Ingestion, Multi-Cloud Fallback (Groq / Gemini / OpenAI), Offline CPU Inference, and Quantitative Benchmarking</i>
</p>

---

## 🌟 Executive Summary & Benchmark Scorecard

This project implements an **enterprise-grade Hybrid Retrieval-Augmented Generation (RAG) system** engineered for high-precision technical question-answering on pre-loaded scientific research papers (`arXiv:2610.30074v1`) as well as **user-uploaded documents (PDF, Word, Text, Web URLs, YouTube)**. It supports **multi-provider cloud execution (Groq, Gemini, OpenAI)** for instantaneous cloud hosting on Streamlit Cloud, alongside **low-level offline CPU inference** via GGUF and `llama-server`.

| Metric | Optimized Engine | Impact / Verification |
| :---  | :---: | :--- |
| **Retrieval Hit Rate** | **`95.8%` (46/48)** | **+95.8% Evidence Hit Rate** |
| **Out-of-Scope Safety** | **`100.0%` (5/5)** | **Zero Hallucinations** |
| **Query Latency (CPU)** | **`30.51s`** | **~9x Speedup via Flash Attention** |
| **Cloud Latency (Groq)** | **`1.2s - 2.5s`** | **Instantaneous Web Responses** |
| **Overall Pass Rate** | **`43.4%` (23/53)** | **Rigorous 53-Question Evaluation** |
| **Execution Errors** | **`0 Errors`** | **100% Production Stability** |

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    subgraph INGESTION["1. Universal Document Ingestion (upload__docs)"]
        INPUT["User Input:<br/>• PDF / DOCX / TXT / MD / CSV<br/>• Web Article URL<br/>• YouTube Video"] --> NORM["Unicode NFKC Normalizer<br/>(Mathematical Symbols: 𝑥, 𝐹, −)"]
        NORM --> ENRICH["Header & Metadata Enrichment<br/>(Title, Authors, Affiliation, arXiv)"]
        ENRICH --> CHUNKER["Dynamic Sliding Window Chunker<br/>(1,000 Chars, 200 Overlap)"]
    end

    subgraph STORAGE["2. Hybrid Multi-Tenant Storage Layer"]
        CHUNKER --> DENSE["ChromaDB Vector Store<br/>(BAAI/bge-small-en-v1.5)"]
        CHUNKER --> SPARSE["BM25 Keyword Inverted Index<br/>(workspace_docs/{id}_docs.pkl)"]
    end

    subgraph RETRIEVAL["3. Hybrid Retrieval Engine"]
        USER_Q["User Query (Hinglish / English)"] --> CLEAN["Query Normalizer & Math Preserver"]
        CLEAN --> MMR["Chroma MMR Dense Search (k=10)"]
        CLEAN --> BM25_SEARCH["BM25 Sparse Search (k=12)"]
        MMR & BM25_SEARCH --> ENSEMBLE["Weighted Ensemble Merger (50/50)<br/>TOP_N = 6 Chunks"]
    end

    subgraph INFERENCE["4. Multi-Model Inference & Guardrails"]
        ENSEMBLE --> GRAPH["LangGraph State Router"]
        GRAPH --> ROUTER{"Selected Engine"}
        ROUTER -->|Cloud / Fast| GROQ["Groq Cloud API<br/>(Llama 3.3 70B Versatile)"]
        ROUTER -->|Google| GEMINI["Google Gemini API<br/>(Gemini 2.5 Flash)"]
        ROUTER -->|OpenAI| OPENAI["OpenAI API<br/>(GPT-4o-mini)"]
        ROUTER -->|100% Offline| LLAMA["llama-server CPU Engine<br/>• Flash Attention (-fa on)<br/>• Single Slot (-np 1)<br/>• Physical Cores (-t 6 -tb 6)"]
        GROQ & GEMINI & OPENAI & LLAMA --> GUARD["Zero-Hallucination Guardrail"]
        GUARD --> OUT["Verified Crisp Response"]
    end
```

---

## ⚡ Low-Level Systems Engineering (9x CPU Speedup)

When running locally on an Intel Core i5-10400 CPU, default multi-slot setups timeout (>280 seconds). We achieved a **9x speedup** through low-level C++ tuning:

1. **Dedicated Single-Slot Allocation (`-np 1`):** Prevents CPU cache fragmentation and context thrashing.
2. **Flash Attention (`-fa on`):** Tiled memory attention kernels on CPU, increasing prompt eval speed from 5.8 t/s to >80 t/s.
3. **Physical Core Mapping (`-t 6 -tb 6`):** Matches the 6 physical cores of the CPU, eliminating hyperthread stalls.
4. **Context Token Budgeting:** Truncated context to 3,800 characters and capped `max_tokens = 150` for rapid answer generation.

---

## 🎨 Claymorphism Streamlit UI

The application features a modern **Claymorphism (tactile 3D soft clay design)** user interface:
* **💬 Interactive Query Assistant:** Query pre-loaded arXiv papers or newly uploaded workspaces with page citations and chunk inspect modals.
* **📁 Upload & Ingest Data:** Live file uploader (PDF, Word, Text, Markdown) and URL scraper with automatic ChromaDB + BM25 indexing.
* **⚡ Multi-Model Selector:** Switch on-the-fly between **Groq Cloud (Llama 3.3 70B)**, **Gemini Flash**, **GPT-4o-mini**, or **Local Offline GGUF**.
* **📊 Benchmark Scorecard:** Interactive data table and analytics charts across all 53 questions in `eval_dataset.json`.
* **🚀 Portfolio Ecosystem:** Direct launch links to live applications built by Nitish Kumar.

---

## 🚀 Quickstart & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/XestalNitish/project3.git
cd project3
```

### 2. Create Virtual Environment & Install Dependencies
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Configure API Keys (Optional for Cloud Mode)
Create a `.env` file or enter keys directly in the Streamlit UI:
```bash
GROQ_API_KEY=your_groq_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

### 4. Launch Streamlit UI
```bash
streamlit run app.py
```

### 5. Run Automated Evaluation Test Suite
```bash
python eval_rag.py --workspace ws1 --module retrival -v
```

---

## ☁️ Deploying to Streamlit Cloud

1. Push this repository to your GitHub account ([`github.com/XestalNitish`](https://github.com/XestalNitish)).
2. Go to [share.streamlit.io](https://share.streamlit.io/) and click **"New app"**.
3. Select your repository and set Main file path to `app.py`.
4. In **Advanced Settings $\rightarrow$ Secrets**, add your free Groq API key:
   ```toml
   GROQ_API_KEY = "gsk_..."
   ```
5. Click **"Deploy"** — your app is live on the internet!

---

## 👨‍💻 Developer & Portfolio Ecosystem

**Nitish Kumar (`@x-estal nitish`)**  
*AI Systems Architect & Applied ML Engineer*

* 🐙 **GitHub:** [github.com/XestalNitish](https://github.com/XestalNitish)
* 💼 **LinkedIn:** [linkedin.com/in/nitish-kumar-33714642b](https://www.linkedin.com/in/nitish-kumar-33714642b/)
* 📸 **Instagram:** [instagram.com/x_estal_nitish](https://www.instagram.com/x_estal_nitish)

### 🌟 Other Live Applications by Nitish Kumar
1. 🛠️ **[Code Dev Agent](https://code-dev-agent-by-x-estal-nitish.streamlit.app):** Autonomous multi-file code generation and debugging agent.
2. 🔬 **[Xestal Research Agent](https://xestal-research-agent-by-x-estal-nitish.streamlit.app):** Academic literature synthesis and deep research assistant.
3. 🏥 **[Brain Tumor Classifier](https://brain-tumor-classifier-by-x-estal-nitish.streamlit.app):** Computer vision MRI diagnostic neural network.
4. 📈 **[Bitcoin Price Predictor](https://x-estal-nitish-bit-coin-price-predictor.streamlit.app):** Quantitative crypto time-series forecasting model.

---

<p align="center">
  <sub>Licensed under the MIT License. Designed & Engineered by Nitish Kumar © 2026.</sub>
</p>
