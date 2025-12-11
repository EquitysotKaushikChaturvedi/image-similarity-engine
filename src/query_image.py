import argparse
import numpy as np
import os
from pathlib import Path
from utils import load_model, compute_image_embedding_from_path, load_index

def main():
    parser = argparse.ArgumentParser(description="Find similar images.")
    parser.add_argument("--query", type=str, required=True, help="Path to query image")
    parser.add_argument("--index_folder", type=str, default="model_data", help="Path to embeddings folder")
    parser.add_argument("--topk", type=int, default=5, help="How many matches to return")
    parser.add_argument("--model", type=str, default="google/siglip-base-patch16-224", help="Model used for indexing")
    
    args = parser.parse_args()
    
    # I need to verify the input file exists before we start anything.
    if not os.path.exists(args.query):
        print(f"Query image '{args.query}' not found.")
        return

    # Load index (embeddings + list of filenames)
    try:
        index_embeddings, index_filenames = load_index(args.index_folder)
    except Exception as e:
        print(f"Could not load index: {e}")
        print("Did you run index_dataset.py first?")
        return

    # Load model just to infer the query (a bit heavy, but simple script)
    model, processor = load_model(args.model)
    
    # Compute query embedding
    query_emb = compute_image_embedding_from_path(args.query, model, processor)
    if query_emb is None:
        print("Failed to compute embedding for query.")
        return

    # Similarity search (Dot product of normalized vectors = Cosine Similarity)
    # query_emb is (D,), index_embeddings is (N, D)
    scores = np.dot(index_embeddings, query_emb)
    
    # Get top-k indices
    # argsort gives smallest first, so we reverse
    top_indices = np.argsort(scores)[::-1][:args.topk]
    
    print(f"\n--- Top {args.topk} Matches ---")
    results = []
    for rank, idx in enumerate(top_indices):
        score = scores[idx]
        fname = index_filenames[idx]
        print(f"{rank+1}. {fname}  (Score: {score:.4f})")
        results.append((fname, score))
        
    # I added this part to generate a simple HTML page.
    # It's much easier to visually verify the results than reading filenames in the console.
    html_content = f"<h2>Query: {Path(args.query).name}</h2><img src='file:///{os.path.abspath(args.query)}' width='200'><br><hr>"
    html_content += "<h3>Matches:</h3>"
    
    dataset_base = Path("data").resolve()
    
    for fname, score in results:
        full_path = dataset_base / fname
        html_content += f"<div><h4>Score: {score:.4f} - {fname}</h4><img src='file:///{full_path}' width='200'></div>"
        
    with open("query_results.html", "w") as f:
        f.write(html_content)
    print("\nTip: Open 'query_results.html' to visualize matches!")

if __name__ == "__main__":
    main()
