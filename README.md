# QinHang-RAG-GeoQA
RAG-based Geological QA System for the Qin-Hang Metallogenic Belt using ChatGLM3-6B and LangChain

# 🔧 Source Code Overview

This directory includes the four main scripts that reproduce the RAG-based geological QA system as described in the paper:
**"Enhancing Geological Knowledge Engineering with Retrieval-Augmented Generation: A Case Study of the Qin–Hang Metallogenic Belt"**

## 🧾 1_pdf_to_txt.py
- Parses geological PDF documents and extracts plain text.
- Supports both OCR-based and standard text-based PDFs.
- Output: `.txt` files used for embedding.

## 🧾 2_txt_to_vector.py
- Converts `.txt` files into vector embeddings using the `bge-large-zh-v1.5` model.
- Embeddings are stored in a FAISS index for fast retrieval.
- Configurable parameters: embedding model, chunk size, dimension, index type.

## 🧾 3_faiss_retrieval.py
- Loads the FAISS vector database and retrieves top-K relevant documents based on user queries.
- Outputs the retrieved context to be passed to the LLM.

## 🧾 4_rag_glm_answer.py
- Performs Retrieval-Augmented Generation (RAG) by feeding question + retrieved context to `ChatGLM3-6B`.
- Generates domain-specific answers with source-backed references.
- Optionally includes support for model fine-tuning or API deployment.

---

## ▶️ Run Instructions

```bash
# 1. Convert PDF corpus to text
python 1_pdf_to_txt.py

# 2. Embed text and build FAISS index
python 2_txt_to_vector.py

# 3. Retrieve context for a sample query
python 3_faiss_retrieval.py --query "What are the mineralization characteristics of the Qin-Hang Belt?"

# 4. Generate response via ChatGLM3-6B
python 4_rag_glm_answer.py


---

```markdown
## 🔄 Reproducibility for Review

To address reviewer concerns regarding evaluation reproducibility and methodological clarity, we provide four core scripts replicating the RAG QA workflow as described in Section 3 of the manuscript.

Each script maps directly to a step in the pipeline: document parsing, semantic embedding, vector-based retrieval, and RAG-based answer generation.

Scripts are fully documented and can be run independently or as a pipeline.


# 📊 Evaluation Design and Results

This folder contains the full quantitative evaluation of our RAG-based geological QA system, as described in Section 4 of the paper.

We present 100 domain-specific questions covering regional geology, mineralization processes, tectonic features, and ore types in the Qin-Hang Metallogenic Belt. Five different models were tested on the same question set:

- ChatGLM3-6B (base)
- ChatGLM3-6B + RAG (ours)
- GPT-4o
- Bing Chat (2024)
- Gemini (Google)
## 🧪 Metric Design

Each answer was scored based on:
- **Precision**: Token overlap with reference
- **Recall**: Token recall against reference
- **F1 Score**: Harmonic mean of precision & recall

All models were evaluated **under identical conditions**, without internet access unless explicitly noted.

## 📌 Sample Entry (from row 1)

```text
Question: What is the Qinhang Mineral Belt?
ChatGLM3-6B F1: 0.8339 | ChatGLM-RAG F1: 0.8838 | GPT-4o F1: 0.8094 | Bing F1: 0.8193 | Gemini F1: 0.8917
```markdown
### ✅ Evaluation Highlights

- 100 QA pairs, across 5 major LLM systems
- RAG-enhanced ChatGLM achieved the **highest average F1 score (0.88)**
- Full dataset, reference answers, and scoring code included for reproducibility
- Both **automated metrics** and **qualitative comments** are presented

## 🔄 RAG QA Pipeline

Our system architecture follows the RAG (Retrieval-Augmented Generation) paradigm, illustrated in the diagram below (Figure 1).

![RAG Pipeline](figures/rag_pipeline_diagram.png)

Each step corresponds to a modular stage in the implementation:

1. **Unstructured Loader**: Parses local documents of various formats (PDF, HTML, JSON, etc.) into text using OCR-enabled extractors.
2. **Text Splitter**: Segments text into logical chunks (e.g., paragraph/sentence-level) using LangChain’s `RecursiveCharacterTextSplitter`.
3. **Text Embedding**: Uses the `bge-large-zh-v1.5` model to convert each chunk into semantic vectors.
4. **Vector Indexing**: Embeddings are stored in FAISS using HNSW or IVF index, depending on data volume.
5. **Query Embedding**: User questions are also embedded and matched using vector similarity.
6. **Prompt Construction**: Retrieved top-K text chunks are used to build context-aware prompts for generation.
7. **Answer Generation**: The `ChatGLM3-6B` model processes the prompt to produce answers.

The pipeline is fully implemented in `src/` and can be run end-to-end with the provided scripts.

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
# 📁 Data Source Overview

The QA system is powered by a local knowledge base composed of 615 documents, including:

- PDF geological reports
- TXT monographs
- JSON metadata
- HTML files from CNKI, Elsevier, VIP databases (non-open-source)
- Manually compiled bilingual glossaries

Due to licensing limitations, the full corpus is not shared. Instead, we provide the structure and 5 sample documents for demonstration.

| File Format | Parser Used             |
|-------------|--------------------------|
| PDF         | `PyMuPDF` + OCR fallback |
| HTML        | `BeautifulSoup` + lxml   |
| JSON        | Python `json` parser     |
| TXT         | UTF-8 parsed             |

Chunking is performed using LangChain’s `RecursiveCharacterTextSplitter`.

# 🧠 Error & Hallucination Analysis

While quantitative metrics (F1, Precision, Recall) provide numeric scores, we also conducted manual inspection on divergent cases.

## Example 1 — Q: "What is the Qinhang Mineral Belt?"

| Model        | Answer Accuracy | Notes |
|--------------|------------------|-------|
| ChatGLM3-6B  | Partial           | Missed several deposit names |
| ChatGLM-RAG  | ✅ Accurate       | Matched with Dexing, Dabaoshan, etc. |
| GPT-4o       | Mostly correct    | Lacked structural explanation |
| Bing         | Partial           | Some hallucinated terms |
| Gemini       | ✅ Good summary   | Included historical/geological context |

## Common Observed Errors

- 🔁 Redundancy: Some models repeat information in different phrasings
- ❌ Hallucinations: Bing occasionally invents tectonic zones
- ⛔ Missing Key Facts: GPT-4o omits key ore deposit names

