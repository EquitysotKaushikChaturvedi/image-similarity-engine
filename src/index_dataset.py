import argparse
import os
from pathlib import Path
from tqdm import tqdm
from utils import load_model, compute_image_embedding_from_path, save_index

# I listed the supported extensions here, but you can add more if needed.
VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

def main():
    parser = argparse.ArgumentParser(description="Index a folder of images for similarity search.")
    parser.add_argument("--dataset", type=str, default="data", help="Path to your images")
    parser.add_argument("--out", type=str, default="model_data", help="Where to save the index")
    parser.add_argument("--model", type=str, default="google/siglip-base-patch16-224", help="Hugging Face model ID")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild index")
    
    args = parser.parse_args()
    
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Error: Dataset folder '{args.dataset}' is missing!")
        return

    # I'm scanning the directory for any files that match our valid extensions.
    all_files = [
        p for p in dataset_path.rglob("*") 
        if p.suffix.lower() in VALID_EXTENSIONS
    ]
    
    if not all_files:
        print("No images found! Put some .jpg or .png files in the dataset folder.")
        return

    print(f"Found {len(all_files)} images. Let's get to work!")

    # Loading the AI model to do the heavy lifting
    model, processor = load_model(args.model)
    
    embeddings = []
    filenames = []
    
    # Process loop
    # I might add batching later if the dataset gets huge (>1000 images), but this is fine for now.
    for image_file in tqdm(all_files, desc="Indexing"):
        emb = compute_image_embedding_from_path(image_file, model, processor)
        if emb is not None:
            embeddings.append(emb)
            # Store relative path to keep it portable
            filenames.append(str(image_file.relative_to(dataset_path)))
            
    if embeddings:
        save_index(embeddings, filenames, args.out)
        print("Done! You can now query your images.")
    else:
        print("Something went wrong, no embeddings were computed.")

if __name__ == "__main__":
    main()
