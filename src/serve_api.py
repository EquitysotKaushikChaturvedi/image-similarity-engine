from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
import uvicorn
import shutil
import os
from pathlib import Path
import tempfile
import numpy as np

# Verify our imports work
try:
    from utils import load_model, load_index, compute_image_embedding_from_path, DEVICE
except ImportError:
    # Fix for running from scripts folder
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from utils import load_model, load_index, compute_image_embedding_from_path, DEVICE

app = FastAPI(title="Image Similarity API", description="Search for images using Deep Learning")

# Enable CORS because frontend devs will thank you later
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
MODEL_NAME = "google/siglip-base-patch16-224"
INDEX_DIR = "model_data"
model = None
processor = None
retrieval_index = None # (embeddings, filenames)

# Mount dataset images so frontend can display them (optional but useful)
if os.path.exists("data"):
    app.mount("/images", StaticFiles(directory="data"), name="images")



@app.on_event("startup")
async def startup_event():
    """Load model and index into memory on startup."""
    global model, processor, retrieval_index
    print("Startup: Loading model and index...")
    model, processor = load_model(MODEL_NAME)
    
    # Check if index exists
    if os.path.exists(INDEX_DIR):
        retrieval_index = load_index(INDEX_DIR)
        print(f"Index loaded! Containing {len(retrieval_index[1])} images.")
    else:
        print("Warning: No index found. You need to run index_dataset.py first!")

@app.post("/search")
async def search(file: UploadFile = File(...), topk: int = 5):
    """
    Upload an image -> get similar images back.
    """
    if retrieval_index is None:
        raise HTTPException(status_code=503, detail="Index not loaded. Server isn't ready.")
        
    index_embeddings, index_filenames = retrieval_index
    
    # Save uploaded file temporarily because our util expects a path
    # (Little bit inefficient but simple to maintain)
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)
        
    try:
        # Compute embedding
        query_emb = compute_image_embedding_from_path(tmp_path, model, processor)
        
        if query_emb is None:
            raise HTTPException(status_code=400, detail="Could not process image.")
            
        # Cosine similarity
        scores = np.dot(index_embeddings, query_emb)
        top_indices = np.argsort(scores)[::-1][:topk]
        
        # I'm formatting the results for JSON response
        matches = []
        for idx in top_indices:
            score = float(scores[idx])
            # User requested strict filtering in the backend too.
            # If it's less than 80% (0.8), we skip it to avoid "bad" results.
            if score < 0.8:
                continue
                
            matches.append({
                "filename": index_filenames[idx],
                "score": score
            })
        
        return {"matches": matches}
        
    finally:
        # Cleanup
        if tmp_path.exists():
            os.remove(tmp_path)

@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE, "index_size": len(retrieval_index[1]) if retrieval_index else 0}

# Mount frontend at the end to avoid shadowing API routes
if os.path.exists("frontend"):
    app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")


if __name__ == "__main__":
    print("Starting server...")
    # Human note: reload=True is great for dev, bad for finding bugs in startup logic sometimes
    uvicorn.run("serve_api:app", host="0.0.0.0", port=8000, reload=False)
