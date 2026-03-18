from fastapi import FastAPI, Query
import torch
from .rag.image_index import VectorStore
from transformers import CLIPProcessor, CLIPModel
import numpy as np

app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)
model.eval()
torch.set_grad_enabled(False)

vector_store = VectorStore()
@app.get("/")
def root():
    return {"status": "ok"}

@app.get("/find_images")
def find_images(
        image_description: str = Query(..., description="Description of image"),
        top_n: int = Query(5, ge=1, le=50)
                ):

    text_inputs = processor(text=[image_description], return_tensors="pt", padding=True).to(device)
    text_embeddings = model.get_text_features(**text_inputs)
    text_embeddings = text_embeddings.detach().cpu().numpy().astype(np.float32)[0]

    res = vector_store.get_image(text_embeddings, top_n=top_n)
    print(res)

    urls = res['ids'][0]

    return {
        "description": image_description,
        "top_n": top_n,
        "urls": urls
    }