import os
import io
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import datasets
import torch
from datasets import load_dataset
from datasets.utils.file_utils import get_datasets_user_agent

from PIL import Image, ImageFile

from transformers import CLIPProcessor, CLIPModel

import torch.nn.functional as F
import numpy as np

from rag.image_index import VectorStore

USER_AGENT = get_datasets_user_agent()
DATASET_SIZE = 60000
ImageFile.LOAD_TRUNCATED_IMAGES = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)
model.eval()
torch.set_grad_enabled(False)

vector_store = VectorStore()

def fetch_single_image(image_url, timeout=5, retries=1):
    last_exc = None
    for _ in range(retries + 1):
        try:
            req = urllib.request.Request(image_url, headers={"user-agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = r.read()

            img = Image.open(io.BytesIO(data))
            img.load()

            # Нормализация: убираем альфу/палитру/странные режимы
            img = img.convert("RGB")

            # Важно: вычищаем info, чтобы не тащить transparency и прочие “кривые” метаданные
            img.info = {}

            # Валидация сериализации: пробуем сохранить в JPEG в память
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=95)
            buf.seek(0)

            # Возвращаем “чистую” картинку, полученную из JPEG-байтов
            clean = Image.open(buf)
            clean.load()
            print("Successfully fetched image {}".format(image_url))
            return clean

        except Exception as e:
            last_exc = e

    print("Unsuccessfully fetched image {}: {}".format(image_url, last_exc))
    return None


def fetch_images(batch, num_threads, timeout=5, retries=1):
    fetch_one = partial(fetch_single_image, timeout=timeout, retries=retries)
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        images = list(executor.map(fetch_one, batch["image_url"]))

    batch["image"] = images
    batch["image_ok"] = [img is not None for img in images]
    return batch


def download_dataset():
    num_threads = 10
    dset = load_dataset("google-research-datasets/conceptual_captions", split="train")
    dset = dset.select(range(DATASET_SIZE))

    dset = dset.map(
        fetch_images,
        batched=True,
        batch_size=50,
        fn_kwargs={"num_threads": num_threads, "timeout": 5, "retries": 1},
    )

    dset = dset.filter(lambda x: x["image_ok"])
    return dset

def calculate_embeddings(batch):
    images = batch["image"]

    image_inputs = processor(images=images, return_tensors="pt").to(device)

    with torch.no_grad():
        image_embeddings = model.get_image_features(**image_inputs)
        image_embeddings = F.normalize(image_embeddings, dim=-1)
    image_embeddings = image_embeddings.detach().cpu().numpy().astype(np.float32)
    batch["image_embeddings"] = [v.tolist() for v in image_embeddings]
    return batch

def store_to_vector_storage(batch):
    embeddings = batch["image_embeddings"]
    image_urls = batch["image_url"]
    vector_store.save_images(embeddings, image_urls)

def main():
    if os.path.exists("./data"):
        dset = datasets.load_from_disk("./data")
        print("Loaded dataset from disk")
    else:
        dset = download_dataset()
        dset.save_to_disk("./data")

    print("Dset loaded")
    print(dset)

    if os.path.exists("./data_with_embeddings"):
        dset = datasets.load_from_disk("./data_with_embeddings")
    else:
        print("CALCULATING EMBEDDINGS")
        dset = dset.map(calculate_embeddings, batched=True, batch_size=64)
        dset.save_to_disk("./data_with_embeddings")

    if not os.path.exists("./chromadb"):
        dset.map(store_to_vector_storage, batched=True, batch_size=1024)

    #text = ["Pair dancing Real photos."]
    #text = ["Bus stop"]
    #text_inputs = processor(text=text, return_tensors="pt", padding=True).to(device)
    #text_embeddings = model.get_text_features(**text_inputs)
    #text_embeddings = text_embeddings.detach().cpu().numpy().astype(np.float32)[0]
    #print(text_embeddings)

    #res = vector_store.get_image(text_embeddings)
    #print(res)

if __name__ == "__main__":
    main()
    print("Done totally")