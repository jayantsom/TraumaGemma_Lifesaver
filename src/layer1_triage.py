# Layer 1: MedSigLIP Polytrauma Triage and Routing

import os
import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, SiglipImageProcessor, SiglipTokenizer

class PolytraumaTriager:
    """
    Zero-shot CT slice triage and anatomical routing using MedSigLIP-448.
    Scores each slice against Brain and Abdomen hemorrhage candidates.
    Routes suspicious slices to their respective Layer 3 U-Nets.
    """

    MODEL_ID = "google/medsiglip-448"
    IMAGE_SIZE = 448

    # Clinically specific and granular labels for Brain and Abdomen CT triage
    LABELS = [
        # Brain - Hemorrhage Positives (Indices 0, 1, 2)
        "CT scan of the head showing subdural or epidural hematoma",
        "CT scan of the head showing subarachnoid or intraparenchymal hemorrhage",
        "CT scan of the head showing midline shift or mass effect from bleeding",
        
        # Brain - Negative (Index 3)
        "Normal CT scan of the head without hemorrhage, mass, or acute abnormality",
        
        # Abdomen - Hemorrhage Positives (Indices 4, 5, 6, 7)
        "CT scan of the abdomen showing liver laceration or perihepatic hematoma",
        "CT scan of the abdomen showing splenic injury or perisplenic fluid",
        "CT scan of the abdomen showing hemoperitoneum, free fluid, or active bleeding",
        "CT scan of the abdomen showing bowel perforation or mesenteric trauma",
        
        # Abdomen - Negative (Index 8)
        "Normal CT scan of the abdomen without hemorrhage, free fluid, or organ injury",
    ]

    # Updating grouping indices for routing logic
    BRAIN_INDICES = [0, 1, 2, 3]
    ABDOMEN_INDICES = [4, 5, 6, 7, 8]
    
    # Grouping all trauma indicators for the threshold gate
    POSITIVE_INDICES = [0, 1, 2, 4, 5, 6, 7]

    def __init__(self, device: str = "cpu", threshold: float = 0.25):
        """
        Initializing the triager on CPU to preserve GPU VRAM for MedGemma.
        """
        self.device = device
        self.threshold = threshold
        self.token = os.environ.get("HF_TOKEN")

        print(f"[PolytraumaTriager] Loading {self.MODEL_ID} on {self.device}...")
        self.image_processor = SiglipImageProcessor.from_pretrained(self.MODEL_ID, token=self.token)
        self.tokenizer = SiglipTokenizer.from_pretrained(self.MODEL_ID, token=self.token)
        self.model = AutoModel.from_pretrained(self.MODEL_ID, token=self.token).to(self.device)
        self.model.eval()
        print(f"[PolytraumaTriager] Ready. Threshold={self.threshold}")

    def score_slice(self, pil_image: Image.Image) -> dict:
        """
        Scoring a single CT slice to determine trauma presence and anatomical region.
        """
        image = pil_image.convert("RGB").resize(
            (self.IMAGE_SIZE, self.IMAGE_SIZE), Image.BILINEAR
        )

        text_inputs = self.tokenizer(
            self.LABELS, padding="max_length", truncation=True, return_tensors="pt"
        ).to(self.device)
        image_inputs = self.image_processor(
            images=image, return_tensors="pt"
        ).to(self.device)
        
        inputs = {**text_inputs, **image_inputs}

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Calculating probabilities
        probs = torch.softmax(outputs.logits_per_image[0], dim=0).cpu().numpy()

        # Determining the anatomical region
        brain_prob = sum(probs[i] for i in self.BRAIN_INDICES)
        abdomen_prob = sum(probs[i] for i in self.ABDOMEN_INDICES)
        region = "brain" if brain_prob > abdomen_prob else "abdomen"

        # Calculating the suspicious score
        suspicious_score = float(sum(probs[i] for i in self.POSITIVE_INDICES))
        top_label = self.LABELS[int(np.argmax(probs))]

        return {
            "region": region,
            "suspicious_score": suspicious_score,
            "suspicious": suspicious_score > self.threshold,
            "top_label": top_label,
        }

    def triage_series(self, pil_images: list, max_slices: int = 5) -> dict:
        """
        Triaging an entire series and returning the most suspicious slices routed by anatomy.
        """
        results = []
        for i, img in enumerate(pil_images):
            result = self.score_slice(img)
            result["slice_index"] = i
            result["image"] = img
            results.append(result)

        # Sorting by most suspicious first
        results.sort(key=lambda x: x["suspicious_score"], reverse=True)

        # Routing to respective dictionaries
        routed_data = {"brain": [], "abdomen": []}
        
        for r in results:
            if r["suspicious"] and len(routed_data[r["region"]]) < max_slices:
                routed_data[r["region"]].append(r)

        # Fallback: If nothing passes the threshold, send the top 1 overall slice
        if not routed_data["brain"] and not routed_data["abdomen"]:
            top_result = results[0]
            routed_data[top_result["region"]].append(top_result)

        return routed_data