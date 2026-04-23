# Downloading preprocessed RSNA abdominal trauma segmentation dataset

# Importing the required libraries
import os
from datasets import load_dataset

# Defining the output directory
output_dir = os.path.join(os.path.dirname(__file__), "abdomen_rsna")

# Creating the directory if it does not exist
os.makedirs(output_dir, exist_ok=True)

# Printing the start status
print("=" * 50)
print("Downloading RSNA Abdominal Trauma - Segmentation Subset")
print("=" * 50)

# Loading the segmentation dataset
dataset = load_dataset(
    "jherng/rsna-2023-abdominal-trauma-detection",
    "segmentation",
    streaming=False,
    trust_remote_code=True
)

# Printing dataset statistics
print("\nDataset loading completed.")
print("Splits:", dataset)
print("Number of training samples:", len(dataset["train"]))

# Saving the dataset to the local disk
print(f"\nSaving to {output_dir}/")
dataset.save_to_disk(output_dir)

# Printing the completion status
print("\nDataset saved successfully to:", output_dir)