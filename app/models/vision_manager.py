import time

import torch
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor

from app.config import DEFAULT_DEVICE, DEFAULT_MODEL_NAME, TOP_K


class VisionManager:
    """Manages a Vision Transformer model for image classification."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str = DEFAULT_DEVICE,
    ) -> None:
        self._model_name = model_name
        self._device = device

        self._processor = ViTImageProcessor.from_pretrained(model_name)
        self._model = ViTForImageClassification.from_pretrained(model_name)
        self._model.to(self._device)
        self._model.eval()

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def device(self) -> str:
        return self._device

    def predict(self, image: Image.Image, top_k: int = TOP_K) -> dict:
        """Run inference on a PIL image and return top-k predictions.

        Args:
            image: A PIL Image to classify.
            top_k: Number of top predictions to return.

        Returns:
            A dict with keys "predictions", "model", and "inference_time_ms".
        """
        inputs = self._processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        start = time.perf_counter()
        with torch.no_grad():
            outputs = self._model(**inputs)
        elapsed_ms = (time.perf_counter() - start) * 1000

        logits = outputs.logits[0]
        probabilities = torch.softmax(logits, dim=-1)
        top_probs, top_indices = torch.topk(probabilities, k=min(top_k, len(probabilities)))

        predictions = [
            {
                "label": self._model.config.id2label[idx.item()],
                "score": round(prob.item(), 4),
            }
            for prob, idx in zip(top_probs, top_indices)
        ]

        return {
            "predictions": predictions,
            "model": self._model_name,
            "inference_time_ms": round(elapsed_ms, 2),
        }
