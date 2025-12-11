import torch
from transformers import AutoProcessor, AutoModel
from PIL import Image
import numpy as np
import os
import pickle
from pathlib import Path

# I'm setting the device automatically here, but you can hardcode 'cuda' if you want specific behavior.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(model_name="google/siglip-base-patch16-224"):
    """
    I'm loading the model and processor from Hugging Face.
    I prefer SigLIP as default, but I made sure CLIP works too.
    """
    print(f"Loading model: {model_name} on {DEVICE}...")
    try:
        # SigLIP and CLIP both work with AutoClasses mainly, but let's be safe
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(DEVICE)
        model.eval() # Don't forget this!
        return model, processor
    except Exception as e:
        print(f"Oops, failed to load {model_name}. Error: {e}")
        raise e

def compute_image_embedding_from_path(image_path, model, processor):
    """
    Reads an image from disk and returns its normalized embedding.
    """
    # Just in case someone passes a string
    if isinstance(image_path, str):
        image_path = Path(image_path)
    
    if not image_path.exists():
        # TODO: Handle missing files more gracefully later?
        print(f"Warning: {image_path} does not exist.")
        return None

    try:
        # Open image and convert to RGB (pngs can be tricky with alpha channels)
        image = Image.open(image_path).convert("RGB")
        
        # Preprocess
        inputs = processor(images=image, return_tensors="pt").to(DEVICE)
        
        # Inference
        with torch.no_grad():
            outputs = model.get_image_features(**inputs) if hasattr(model, 'get_image_features') else model.get_text_features(**inputs) 
            # Wait, get_image_features is for CLIP/SigLIP vision models. 
            # If it's a pure vision model, 'outputs.pooler_output' or similar might be needed if not using the bespoke methods.
            # But for SigLIP/CLIP `get_image_features` is standard in HF transformers.
        
        # Normalize (L2) - crucial for cosine similarity!
        embedding = outputs / outputs.norm(p=2, dim=-1, keepdim=True)
        return embedding.cpu().numpy().flatten()
        
    except Exception as e:
        print(f"Error processing {image_path.name}: {e}")
        return None

def save_index(embeddings: list, filenames: list, out_dir: str):
    """
    Saves embeddings as .npy and filenames as .pkl
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Stack list of arrays into one big matrix
    emb_array = np.vstack(embeddings).astype('float32') # float32 is usually enough
    
    np.save(out_path / "embeddings.npy", emb_array)
    with open(out_path / "filenames.pkl", 'wb') as f:
        pickle.dump(filenames, f)
    
    print(f"Saved index to {out_dir} (Shape: {emb_array.shape})")

def load_index(out_dir: str):
    """
    Loads embeddings and filenames.
    """
    # Using os.path here just to mix it up a bit (valid python style!)
    if not os.path.exists(out_dir):
        raise FileNotFoundError(f"Index folder '{out_dir}' not found.")
        
    emb_path = os.path.join(out_dir, "embeddings.npy")
    names_path = os.path.join(out_dir, "filenames.pkl")
    
    embeddings = np.load(emb_path)
    with open(names_path, 'rb') as f:
        filenames = pickle.load(f)
        
    return embeddings, filenames
