# Downloading and extracting the brain hemorrhage dataset directly

# Importing the required libraries
import os
import urllib.request
import zipfile

# Defining the output directory
output_dir = os.path.join(os.path.dirname(__file__), "brain_segmentation")

# Creating the directory if it does not exist
os.makedirs(output_dir, exist_ok=True)

# Defining the direct download URL and local zip path
url = "https://huggingface.co/datasets/WuBiao/BHSD/resolve/main/label_192.zip"
zip_path = os.path.join(output_dir, "label_192.zip")

# Printing the start status
print("=" * 50)
print("Downloading Brain Hemorrhage Segmentation Dataset")
print("=" * 50)

# Downloading the zip file directly
print("\nDownloading the zip file...")
urllib.request.urlretrieve(url, zip_path)

# Extracting the zip file
print("\nExtracting the contents...")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(output_dir)

# Removing the downloaded zip file to save space
print("\nCleaning up the zip file...")
os.remove(zip_path)

# Printing the completion status
print("\nDataset downloaded and extracted successfully to:", output_dir)