import os
import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, SiglipImageProcessor, SiglipTokenizer

from config import (
    MEDSIGLIP_MODEL,
    TRIAGE_THRESHOLD,
    ABDOMEN_LABELS,
    HEAD_LABELS,
    ABDOMEN_POSITIVE_IDX,
    HEAD_POSITIVE_IDX,
)

class MedSigLIPTriager:
    """
    Zero-shot CT slice triage for abdomen or head.
    Scores each slice against domain‑specific text candidates.
    Returns suspicious slices for downstream MedGemma.
    """

    MODEL_ID = MEDSIGLIP_MODEL
    IMAGE_SIZE = 448

    def __init__(self, domain="abdomen", device="cpu", threshold=None, hf_token=None):
        """
        domain: "abdomen" or "head"
        device: "cpu" or "cuda"
        threshold: suspicious score cutoff (default from config)
        hf_token: optional HF token (falls back to HF_TOKEN env var)
        """
        self.domain = domain
        self.device = device
        self.threshold = threshold if threshold is not None else TRIAGE_THRESHOLD

        # Select labels and positive indices based on domain
        if domain == "abdomen":
            self.labels = ABDOMEN_LABELS
            self.positive_indices = ABDOMEN_POSITIVE_IDX
        else:   # head
            self.labels = HEAD_LABELS
            self.positive_indices = HEAD_POSITIVE_IDX

        token = hf_token or os.environ.get("HF_TOKEN")
        print(f"[MedSigLIPTriager] Loading {self.MODEL_ID} for {domain} on {self.device}...")
        self.image_processor = SiglipImageProcessor.from_pretrained(self.MODEL_ID, token=token)
        self.tokenizer = SiglipTokenizer.from_pretrained(self.MODEL_ID, token=token)
        self.model = AutoModel.from_pretrained(self.MODEL_ID, token=token).to(self.device)
        self.model.eval()
        print(f"[MedSigLIPTriager] Ready. Threshold={self.threshold}")

    def score_slice(self, pil_image):
        """Return dict with suspicious_score, suspicious bool, per‑label probs."""
        # Resize and convert to RGB (SigLIP expects RGB)
        image = pil_image.convert("RGB").resize((self.IMAGE_SIZE, self.IMAGE_SIZE), Image.BILINEAR)

        text_inputs = self.tokenizer(
            self.labels,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        image_inputs = self.image_processor(
            images=image,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**{**text_inputs, **image_inputs})

        probs = torch.softmax(outputs.logits_per_image[0], dim=0).cpu().numpy()
        scores = {label: float(probs[i]) for i, label in enumerate(self.labels)}
        suspicious_score = float(sum(probs[i] for i in self.positive_indices))
        top_label = self.labels[int(np.argmax(probs))]

        return {
            "scores": scores,
            "suspicious_score": suspicious_score,
            "suspicious": suspicious_score > self.threshold,
            "top_label": top_label,
        }

    def triage_slices(self, pil_images):
        """Score all slices, return list sorted by suspicious_score descending."""
        results = []
        for i, img in enumerate(pil_images):
            res = self.score_slice(img)
            res["slice_index"] = i
            results.append(res)
        return sorted(results, key=lambda x: x["suspicious_score"], reverse=True)

    def get_suspicious_slices(self, pil_images, max_slices=5):
        """Return (suspicious_images_list, all_results). Always at least 1 slice."""
        all_results = self.triage_slices(pil_images)
        suspicious = [r for r in all_results if r["suspicious"]]
        if not suspicious:
            suspicious = [all_results[0]]
        top = suspicious[:max_slices]
        indices = [r["slice_index"] for r in top]
        suspicious_images = [pil_images[i] for i in indices]
        return suspicious_images, all_results

    def get_triage_summary(self, all_results):
        """Return summary dict for UI."""
        by_index = sorted(all_results, key=lambda x: x["slice_index"])
        per_slice_scores = [r["suspicious_score"] for r in by_index]
        suspicious_count = sum(1 for r in all_results if r["suspicious"])
        return {
            "total_slices": len(all_results),
            "suspicious_count": suspicious_count,
            "max_score": float(max(per_slice_scores)) if per_slice_scores else 0.0,
            "mean_score": float(np.mean(per_slice_scores)) if per_slice_scores else 0.0,
            "per_slice_scores": per_slice_scores,
        }