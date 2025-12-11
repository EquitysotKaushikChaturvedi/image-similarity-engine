import os
import shutil
from pathlib import Path

def cleanup():
    root = Path(".")
    
    # Remove __pycache__ directories
    for p in root.rglob("__pycache__"):
        print(f"Removing {p}")
        shutil.rmtree(p)
        
    # Remove .DS_Store (Mac junk on Windows sometimes)
    for p in root.rglob(".DS_Store"):
        print(f"Removing {p}")
        os.remove(p)
        
    # Remove any stray .tmp files
    for p in root.rglob("*.tmp"):
        print(f"Removing {p}")
        os.remove(p)

    print("Cleanup complete.")

if __name__ == "__main__":
    cleanup()
