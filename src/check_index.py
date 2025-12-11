import numpy as np
import pickle
import os
import sys

def check(index_dir="embeddings"):
    print(f"Checking index in '{index_dir}'...")
    
    emb_path = os.path.join(index_dir, "embeddings.npy")
    names_path = os.path.join(index_dir, "filenames.pkl")
    
    if not os.path.exists(emb_path) or not os.path.exists(names_path):
        print("Missing index files. Run index_dataset.py first.")
        sys.exit(1)
        
    embs = np.load(emb_path)
    with open(names_path, 'rb') as f:
        names = pickle.load(f)
        
    print(f"Embeddings shape: {embs.shape}")
    print(f"Number of filenames: {len(names)}")
    
    if embs.shape[0] != len(names):
        print(" CRITICAL ERROR: Mismatch between embeddings count and filenames count!")
        sys.exit(1)
        
    if embs.shape[1] < 10:
        print(" Warning: Embedding dimension seems suspiciously small.")
        
    print(" Index looks broken... wait, I mean consistent! Good job.")

if __name__ == "__main__":
    # Allow passing dir as arg
    dir_to_check = sys.argv[1] if len(sys.argv) > 1 else "embeddings"
    check(dir_to_check)
