# Image Similarity Search Project

Hi! This is a complete, custom-built image similarity search engine. I designed it to be simple, fast, and easy to extend. It uses state-of-the-art AI (Hugging Face SigLIP) to "see" your images and find matches.

## Project Structure

I've organized the project to be clean and production-ready:

*   **`src/`**: Contains all the Python source code.
    *   `index_dataset.py`: The script to scan your data and build the search index.
    *   `serve_api.py`: The backend API server (FastAPI).
    *   `query_image.py`: A command-line tool to test searches.
    *   `utils.py`: Core logic for model loading and math.
*   **`data/`**: Put your images here! (e.g., `.jpg`, `.png`).
*   **`model_data/`**: Where the AI stores its "memory" (the index files).
*   **`frontend/`**: The web interface (HTML/CSS/JS).
*   **`notebooks/`**: Contains `train_custom_index.ipynb` for interactive experiments.

## How to Run It

### 1. Installation

First, make sure you have Python installed. Then grab the dependencies:

```bash
pip install -r requirements.txt
```

### 2. Add Your Data

Put your images into the `data/` folder.
*(If you're just testing, I included a script to download demo images: `python src/download_demo.py`)*

### 3. Build the Index

Run this command to let the AI learn your images:

```bash
python src/index_dataset.py
```

It will scan `data/` and save the index to `model_data/`.

### 4. Start the Web Interface

Launch the server:

```bash
python src/serve_api.py
```

Now open your browser to: **http://127.0.0.1:8000**

Upload an image and see the magic happen!
I've set it to only show **high-confidence matches (>80%)** so you don't get junk results.

## Tech Stack

*   **Backend**: Python, FastAPI
*   **AI Model**: `google/siglip-base-patch16-224` (Hugging Face)
*   **Frontend**: Vanilla HTML/JS (Clean & Fast)
*   **Index**: Simple NumPy/Pickle storage (no complex database required)

Enjoy! Let me know if you have questions.
