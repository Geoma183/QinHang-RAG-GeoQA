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

