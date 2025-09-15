import os
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import logging
import argparse
import traceback

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ===================== Utility Functions =====================

def load_faiss_index(index_path):
    if os.path.exists(index_path):
        index = faiss.read_index(index_path)
        logger.info(f"Loaded FAISS index from: {index_path}")
        return index
    else:
        raise FileNotFoundError(f"Index not found: {index_path}")

def load_text_mapping(pkl_path):
    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            text_map = pickle.load(f)
        logger.info(f"Loaded text mapping from: {pkl_path}")
        return text_map
    else:
        raise FileNotFoundError(f"Text map not found: {pkl_path}")

def encode_query(model, query):
    embedding = model.encode([query], convert_to_tensor=True).cpu().numpy()
    embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
    return embedding

def search_indexes(query_vector, indexes, text_maps, top_k=10, weights=None):
    if weights is None:
        weights = [1.0] * len(indexes)

    all_results = []
    for idx, (index, text_map, weight) in enumerate(zip(indexes, text_maps, weights)):
        distances, indices = index.search(query_vector, top_k)
        for dist, idx_val in zip(distances[0], indices[0]):
            if idx_val < len(text_map):
                all_results.append({
                    "text": text_map[idx_val],
                    "distance": dist,
                    "weight": weight,
                    "weighted_score": dist * weight,
                    "source": f"Index-{idx + 1}"
                })

    all_results = sorted(all_results, key=lambda x: x["weighted_score"], reverse=True)
    return all_results[:top_k]

def display_results(results, max_display=10):
    for i, res in enumerate(results[:max_display], 1):
        print(f"\n[Result {i}]")
        print(f"Score: {res['distance']:.4f} (Weighted: {res['weighted_score']:.4f})")
        print(f"Source: {res['source']}")
        print(f"Text: {res['text']}\n")

# ===================== Main Script =====================

def parse_args():
    parser = argparse.ArgumentParser(description="Query multiple FAISS indices for semantic search")
    parser.add_argument('--core_index', required=True, help="Path to core FAISS index file")
    parser.add_argument('--core_map', required=True, help="Path to core text map (pickle)")
    parser.add_argument('--extra_index', required=True, help="Path to additional FAISS index file")
    parser.add_argument('--extra_map', required=True, help="Path to additional text map (pickle)")
    parser.add_argument('--model_name', default='paraphrase-multilingual-MiniLM-L12-v2', help="Name of the SentenceTransformer model")
    parser.add_argument('--top_k', type=int, default=10, help="Number of top results to return")
    return parser.parse_args()

def main():
    args = parse_args()
    try:
        logger.info("Loading model...")
        model = SentenceTransformer(args.model_name)

        logger.info("Loading indices and mappings...")
        core_index = load_faiss_index(args.core_index)
        core_map = load_text_mapping(args.core_map)
        extra_index = load_faiss_index(args.extra_index)
        extra_map = load_text_mapping(args.extra_map)

        indexes = [core_index, extra_index]
        text_maps = [core_map, extra_map]
        weights = [1.0, 1.0]  # Modify if you want to give priority to certain sources

        query = input("Enter your query: ").strip()
        if not query:
            print("Empty query. Exiting.")
            return

        logger.info("Encoding query...")
        query_vector = encode_query(model, query)

        logger.info("Searching across indices...")
        results = search_indexes(query_vector, indexes, text_maps, top_k=args.top_k, weights=weights)

        display_results(results)

    except Exception as e:
        logger.error(f"Search failed: {e}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
