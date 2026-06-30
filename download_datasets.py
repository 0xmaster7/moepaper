import os
import shutil
from huggingface_hub import hf_hub_download, HfApi

def download_datasets():
    print("Welcome! Let's download your public datasets from Hugging Face.\n")
    
    datasets_to_download = [
        {"repo_id": "0xmaster7/merged.csv", "filename": "merged.csv"},
        {"repo_id": "0xmaster7/benign_ember_feaures.csv", "filename": "benign_ember_features.csv"}
    ]
    
    api = HfApi()

    for ds in datasets_to_download:
        repo_id = ds["repo_id"]
        filename = ds["filename"]
        destination = os.path.join(os.getcwd(), filename)
        
        print(f"\nChecking {filename} in {repo_id}...")
        
        try:
            # First, verify we have access to the repository (doesn't download the massive file)
            api.dataset_info(repo_id)
            
            # If the file already exists on your system, skip the download!
            if os.path.exists(destination):
                print(f"✅ Connection successful! '{filename}' already exists locally.")
                print("Skipping download to save bandwidth and time. (It works!)")
                continue
            
            print(f"Downloading {filename}...")
            # Download the file to a cache folder
            cached_path = hf_hub_download(
                repo_id=repo_id, 
                filename=filename, 
                repo_type="dataset"
            )
            
            # Move the downloaded file to the current working directory
            shutil.copy2(cached_path, destination)
            print(f"✅ Successfully downloaded and saved to: {destination}")
            
        except Exception as e:
            print(f"❌ Failed to access or download {filename}.")
            print(f"Error details: {e}")
            print("Please check if the dataset is Public on Hugging Face and the name matches exactly.")

if __name__ == "__main__":
    download_datasets()
