from transformers import CLIPProcessor, CLIPModel
import torch
import numpy as np

class Embedder():
    def __init__(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self._processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)
        self._model.eval()
        torch.set_grad_enabled(False)

    def embed_text(self, text):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        text_inputs = self._processor(text=[text], return_tensors="pt", padding=True).to(device)
        text_embeddings = self._model.get_text_features(**text_inputs)
        text_embeddings = text_embeddings.detach().cpu().numpy().astype(np.float32)[0]
        return text_embeddings
