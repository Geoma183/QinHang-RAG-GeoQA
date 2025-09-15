# 🔍 QinHang-RAG-GeoQA

**Retrieval-Augmented Geological Question Answering System for the Qin–Hang Metallogenic Belt**  
Using ChatGLM3-6B, LangChain, and FAISS

This repository supports the reproducibility of the research paper:  
**_Enhancing Geological Knowledge Engineering with Retrieval-Augmented Generation: A Case Study of the Qin–Hang Metallogenic Belt_**

---

## 📦 1. Project Overview

This project introduces a domain-specific RAG (Retrieval-Augmented Generation) pipeline for intelligent geological Q&A.  
Key components:

- ✅ Modular RAG pipeline built with LangChain
- ✅ Embedding via `bge-large-zh-v1.5` and FAISS indexing
- ✅ 615-document bilingual geological knowledge base
- ✅ Evaluation across 5 LLMs (ChatGLM, GPT-4o, Bing, Gemini)
- ✅ Human error analysis + full traceability

---

## 🔄 2. System Architecture

The system follows the standard RAG pipeline. See the workflow below:

![RAG Pipeline](figures/rag_pipeline_diagram.png)

### 🔁 Step-by-step:

1. **Document Parsing**: OCR-enabled loader for PDF, TXT, HTML, JSON, etc.
2. **Text Chunking**: `RecursiveCharacterTextSplitter` ensures semantic coherence
3. **Embedding**: Each chunk is embedded with `bge-large-zh-v1.5`
4. **Vector Indexing**: Stored in FAISS using HNSW or IVF index
5. **Query Embedding**: User questions embedded into vector space
6. **Similarity Search**: Top-K relevant chunks retrieved
7. **Prompting**: Question + Context used to form final prompt
8. **Answer Generation**: Handled by `ChatGLM3-6B` for contextual responses

```yaml
# 🔧 Sample Configuration (configs/rag_config.yaml)
embedding_model: bge-large-zh-v1.5
chunk_size: 500
chunk_overlap: 50
vector_index:
  type: HNSW
  dimension: 1024
  metric: cosine
retriever:
  top_k: 5
llm:
  model: ChatGLM3-6B
  inference_mode: local  # or api
💾 3. Data Collection & Preprocessing

A total of 615 research documents on the Qin–Hang Metallogenic Belt were collected:

Source Platform	Number
CNKI (China National Knowledge Infrastructure)	213
Google Scholar	240
VIP Database	7
MDPI	27
Elsevier Journals	118
Springer Journals	5
Specialized Monographs	5
Total	615
📄 Formats:

PDF, HTML, TXT, JSON, CSV

⚙️ Processing Workflow:

OCR parsing via LangChain loaders

Cleaned and chunked (500 tokens, 50 overlap)

Embedded with bge-large-zh-v1.5

Indexed using FAISS

📁 Due to licensing, only 5 sample documents are shared.

🧠 4. Source Code & Usage

All source code is located in /src
.

🧾 Scripts
File	Description
1_pdf_to_txt.py	Parses PDFs into .txt
2_txt_to_vector.py	Embeds text and stores in FAISS
3_faiss_retrieval.py	Retrieves top-K chunks for query
4_rag_glm_answer.py	RAG generation using ChatGLM3-6B
▶️ Run Example
# 1. Convert PDF corpus to plain text
python 1_pdf_to_txt.py

# 2. Create vector database from text
python 2_txt_to_vector.py

# 3. Retrieve similar context from vector store
python 3_faiss_retrieval.py --query "What are the mineralization characteristics of the Qin-Hang Belt?"

# 4. Generate answer using ChatGLM3-6B
python 4_rag_glm_answer.py

📊 5. Evaluation
🔬 Dataset

We constructed a 100-question test set covering:

Regional geology

Ore types

Tectonic evolution

Deposit examples

🤖 Models Compared

ChatGLM3-6B (baseline)

ChatGLM3-6B + RAG (our method)

GPT-4o

Bing Chat (2024)

Gemini (Google)

🧪 Metrics

Precision: Overlap with reference tokens

Recall: Coverage of reference by response

F1 Score: Harmonic mean of the above

📌 All models evaluated under identical conditions, with internet access disabled unless stated.
Question: What is the Qinhang Mineral Belt?
ChatGLM3-6B F1: 0.8339 | ChatGLM-RAG F1: 0.8838 | GPT-4o F1: 0.8094 | Bing F1: 0.8193 | Gemini F1: 0.8917
✅ Highlights

100 QA pairs × 5 models × 3 metrics

RAG-enhanced ChatGLM achieved highest F1 (0.8838)

Full table at: evaluation/qa_evaluation_table.csv

Scripts and references included



🧐 6. Error & Hallucination Analysis

We manually reviewed performance across multiple questions.

Example Case
Model	Verdict	Notes
ChatGLM3-6B	❌ Partial	Missed key deposit names
ChatGLM-RAG	✅ Accurate	Mentioned Dexing, Dabaoshan, etc.
GPT-4o	⚠️ Incomplete	Lacked structural geology
Bing	❌ Hallucinated	Made up tectonic terms
Gemini	✅ Good summary	Included historical/geological framing
Common Issues

🔁 Repetition

❌ Hallucinated place names

⛔ Omission of key terms

Full discussion in evaluation/error_analysis.md



📚 7. For Reviewers & Reproducibility

This repository is designed to support peer review and academic scrutiny.

✅ Pipeline transparency (code/scripts/configs)

✅ Data origin clarity (615 documents)

✅ Evaluation with automated + manual metrics

✅ Structured folder system

✅ Open diagrams & prompt templates

Please feel free to explore, run, or adapt this system for geoscientific question answering.
