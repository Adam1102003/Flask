import os
import gdown

# روابط Google Drive IDs
files = {
    "inception_classifier.keras": "1rDCVJnMCaa2Wzet7qik1fD7oH6vQ74zN",
    "Puzzle_data.zip": "1KuqdLjy3cN8xlV15EcCPcxTLt4-kPLAh"
}

for filename, file_id in files.items():
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, filename, quiet=False)
    else:
        print(f"{filename} already exists. Skipping download.")
