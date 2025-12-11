import os
import requests
from pathlib import Path
from tqdm import tqdm

# We will download 200 random images for a larger dataset
NUM_IMAGES = 200

def download_demo_dataset():
    """
    Downloads 200 random images to 'dataset_images/' folder.
    """
    save_dir = Path("data")
    save_dir.mkdir(exist_ok=True)
    
    existing = len(list(save_dir.glob("*.[jJ][pP][gG]")))
    if existing >= NUM_IMAGES:
        print(f"Dataset already has {existing} images. Skipping download.")
        return

    print(f"Downloading {NUM_IMAGES} images...")
    
    for i in tqdm(range(1, NUM_IMAGES + 1)):
        filename = f"image_{i:03d}.jpg"
        filepath = save_dir / filename
        
        if filepath.exists():
            continue
            
        # Use Picsum for reliable random images with seeds to ensure consistency if re-run
        url = f"https://picsum.photos/seed/{i}/600/600"
        
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                with open(filepath, 'wb') as f:
                    f.write(response.content)
            else:
                print(f"Failed to download {url}")
        except Exception as e:
            print(f"Error downloading {filename}: {e}")

if __name__ == "__main__":
    download_demo_dataset()
            
    print("Demo dataset ready!")
    print("Run: python scripts/index_dataset.py --dataset dataset_images")
