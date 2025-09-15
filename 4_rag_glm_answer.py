import os
import torch
import faiss
import pickle
import logging
import traceback
import numpy as np
import argparse

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer

# ========== Logging Configuration ==========
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ========== Argument Parsing ==========
def parse_args():
    parser = argparse.ArgumentParser(description="GLM-based RAG QA with FAISS retrieval")
    parser.add_argument("--glm_model_path", required=True, help="Path to GLM model directory")
    parser.add_argument("--core_index", required=True, help="Path to core FAISS index directory")
    parser.add_argument("--core_map", required=True, help="Path to core index pickle file")
    parser.add_argument("--extra_index", required=True, help="Path to additional FAISS index directory")
    parser.add_argument("--extra_map", required=True, help="Path to additional index pickle file")
    parser.add_argument("--top_k", type=int, default=10, help="Number of results to retrieve from each index")
    parser.add_argument("--context_k", type=int, default=3, help="Number of top passages used for answering")
    parser.add_argument("--weights", nargs=2, type=float, default=[1.5, 1.0], help="Weights for core and additional indexes")
    return parser.parse_args()

# ========== Model Loaders ==========
def load_glm_model(path):
    logger.info("Loading GLM-4 model...")
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    model = AutoModelForCausalLM.from_pretrained(
        path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    model.gradient_checkpointing_enable()

    if torch.cuda.device_count() > 1:
        logger.info(f"{torch.cuda.device_count()} GPUs detected. Using DataParallel.")
        model = torch.nn.DataParallel(model)
    else:
        logger.info("Single GPU detected.")

    return model, tokenizer

def load_retrieval_model():
    logger.info("Loading retrieval model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device=device)
    return model

# ========== FAISS Index Loaders ==========
def load_faiss_index(index_dir, pkl_path):
    index_path = os.path.join(index_dir, "index.faiss")
    index = faiss.read_index(index_path)
    with open(pkl_path, "rb") as f:
        text_map = pickle.load(f)
    logger.info(f"Loaded FAISS index and mapping from: {index_dir}")
    return index, text_map

# ========== Retrieval ==========
def search_indexes(query_vector, indexes, text_maps, weights, top_k):
    results = []
    for idx, (index, mapping, weight) in enumerate(zip(indexes, text_maps, weights)):
        logger.info(f"Searching index {idx+1} with weight {weight}")
        distances, indices = index.search(query_vector, top_k)
        for dist, ind in zip(distances[0], indices[0]):
            if ind != -1 and ind < len(mapping):
                results.append({
                    "text": mapping[ind],
                    "distance": dist,
                    "weighted_distance": dist * weight,
                    "source": f"Index {idx+1}"
                })

    results = sorted(results, key=lambda x: x["weighted_distance"])[:top_k]
    return results

# ========== Answer Generation ==========
def generate_answer_with_glm(question, context, model, tokenizer, max_length=300):
    prompt = f"""Based on the following documents, answer the question below:

{context}

Question: {question}
Answer:"""

    try:
        device = next(model.parameters()).device
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        if isinstance(model, torch.nn.DataParallel):
            generate_fn = model.module.generate
        else:
            generate_fn = model.generate

        output = generate_fn(
            **inputs,
            max_new_tokens=max_length,
            temperature=0.7,
            top_p=0.85,
            top_k=40,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )

        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
        if "Answer:" in decoded:
            return decoded.split("Answer:")[-1].strip()
        return decoded.strip()
    except Exception as e:
        logger.error(f"Answer generation failed: {e}")
        return "Sorry, I couldn't generate an answer."

# ========== Main ==========
def main():
    args = parse_args()

    # Load models
    retrieval_model = load_retrieval_model()
    glm_model, glm_tokenizer = load_glm_model(args.glm_model_path)

    # Load FAISS indexes
    core_index, core_map = load_faiss_index(args.core_index, args.core_map)
    extra_index, extra_map = load_faiss_index(args.extra_index, args.extra_map)

    indexes = [core_index, extra_index]
    maps = [core_map, extra_map]

    # Input
    question = input("Enter your query: ").strip()
    if not question:
        logger.info("Empty query received. Exiting.")
        return

    logger.info("Encoding query...")
    query_vector = retrieval_model.encode(question, convert_to_tensor=True).cpu().numpy().reshape(1, -1)

    logger.info("Searching indexes...")
    results = search_indexes(query_vector, indexes, maps, args.weights, args.top_k)

    if not results:
        print("No relevant documents found.")
        return

    # Prepare context
    context_docs = results[:args.context_k]
    context = "\n\n".join([f"Doc {i+1}:\n{res['text']}" for i, res in enumerate(context_docs)])

    # Generate and print answer
    logger.info("Generating answer...")
    answer = generate_answer_with_glm(question, context, glm_model, glm_tokenizer)

    print("\n===== Answer =====")
    print(answer)

if __name__ == "__main__":
    main()
