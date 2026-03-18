from transformers import CLIPProcessor, CLIPModel
import torch
import numpy as np
import torch.nn.functional as F

class Embedder():
    def __init__(self):
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self._device )
        self._processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)
        self._model.eval()
        torch.set_grad_enabled(False)

    def embed_text(self, text):
        text_inputs = self._processor(text=[text], return_tensors="pt", padding=True).to(self._device )
        text_embeddings = self._model.get_text_features(**text_inputs)
        text_embeddings = text_embeddings.detach().cpu().numpy().astype(np.float32)[0]
        return text_embeddings

    def embed_images(self, images):
        image_inputs = self._processor(images=images, return_tensors="pt").to(self._device )

        with torch.no_grad():
            image_embeddings = self._model.get_image_features(**image_inputs)
            image_embeddings = F.normalize(image_embeddings, dim=-1)
        image_embeddings = image_embeddings.detach().cpu().numpy().astype(np.float32)

        return [v.tolist() for v in image_embeddings]