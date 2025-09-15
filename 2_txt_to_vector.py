import os
import torch
import numpy as np
import json
import argparse
from sentence_transformers import SentenceTransformer
import faiss
import warnings
from tqdm import tqdm
import logging
from logging.handlers import RotatingFileHandler
import traceback

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        RotatingFileHandler("vectorization.log", maxBytes=10**7, backupCount=3),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
RETRIEVAL_MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'

# ========================== Vectorizer ==========================
class Vectorizer:
    def __init__(self, model_name, input_folder, index_folder, record_path):
        self.model_name = model_name
        self.input_folder = input_folder
        self.index_folder = index_folder
        self.record_path = record_path
        self.embedding_model = self.load_model()

    def load_model(self):
        """Load embedding model"""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading model on device: {device}")
        try:
            model = SentenceTransformer(self.model_name, device=device)
            _ = model.encode("test text", convert_to_tensor=False)
            logger.info(f"Model loaded successfully: {self.model_name}")
            return model
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            raise

    def load_processed_files(self):
        """Read record of already processed files"""
        if os.path.exists(self.record_path):
            try:
                with open(self.record_path, 'r', encoding='utf-8') as f:
                    return set(json.load(f))
            except:
                logger.warning("Invalid record file. Starting fresh.")
                return set()
        return set()

    def update_processed_files(self, new_files):
        """Update record with newly processed files"""
        processed = self.load_processed_files()
        processed.update(new_files)
        with open(self.record_path, 'w', encoding='utf-8') as f:
            json.dump(list(processed), f, ensure_ascii=False, indent=2)
        logger.info(f"Updated processed record with {len(new_files)} files.")

    def load_chunks(self, processed_files):
        """Load chunked JSON text"""
        documents = []
        all_files = [f for f in os.listdir(self.input_folder) if f.endswith('.json')]
        new_files = [f for f in all_files if f not in processed_files]

        logger.info(f"Found {len(new_files)} new files to vectorize.")

        for fname in tqdm(new_files, desc="Loading text chunks"):
            fpath = os.path.join(self.input_folder, fname)
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    chunks = data.get("text_chunks", [])
                    if isinstance(chunks, list):
                        documents.extend(chunks)
            except Exception as e:
                logger.warning(f"Failed to load {fname}: {e}")
        return documents, new_files

    def normalize_embeddings(self, embeddings):
        """L2 normalize vectors"""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / norms

    def vectorize_and_store(self, documents, batch_size=100):
        """Convert text to vectors and store in FAISS"""
        os.makedirs(self.index_folder, exist_ok=True)
        index_path = os.path.join(self.index_folder, "index.faiss")

        if os.path.exists(index_path):
            index = faiss.read_index(index_path)
            logger.info("Loaded existing FAISS index.")
        else:
            index = None

        for i in tqdm(range(0, len(documents), batch_size), desc="Vectorizing"):
            batch = documents[i:i+batch_size]
            embeddings = self.embedding_model.encode(batch, convert_to_tensor=False)
            normed = self.normalize_embeddings(embeddings)

            if index is None:
                dim = normed.shape[1]
                index = faiss.IndexFlatIP(dim)
                logger.info(f"Created new FAISS index with dimension: {dim}")

            index.add(normed)

        faiss.write_index(index, index_path)
        logger.info(f"FAISS index saved to: {index_path}")

# ========================== Utility ==========================
def parse_args():
    parser = argparse.ArgumentParser(description="Vectorize chunked text and store in FAISS index.")
    parser.add_argument('--input_folder', required=True, help="Path to folder with chunked JSON files")
    parser.add_argument('--index_folder', required=True, help="Path to save FAISS index")
    parser.add_argument('--record_path', default="outputs/vectorization_record.json", help="Path to processed record file")
    return parser.parse_args()

# ========================== Main ==========================
def main():
    args = parse_args()

    try:
        vectorizer = Vectorizer(
            model_name=RETRIEVAL_MODEL_NAME,
            input_folder=args.input_folder,
            index_folder=args.index_folder,
            record_path=args.record_path
        )

        processed = vectorizer.load_processed_files()
        documents, new_files = vectorizer.load_chunks(processed)

        if not documents:
            logger.info("No new documents found.")
            return

        vectorizer.vectorize_and_store(documents)
        vectorizer.update_processed_files(new_files)

    except Exception as e:
        logger.error(f"Error during vectorization: {e}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
