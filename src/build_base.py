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

USER_AGENT = get_datasets_user_agent()
DATASET_SIZE = 60000
ImageFile.LOAD_TRUNCATED_IMAGES = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)

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

def calculate_embeddings(row):
    image_inputs = processor(images=row["image"], return_tensors="pt").to(device)

    #text = ["Mounts", "Bus station", "Woman in skirt"]

    #text_inputs = processor(text=text, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        image_embeddings = model.get_image_features(**image_inputs)
        image_embeddings = F.normalize(image_embeddings, dim=-1)[0]
        print(image_embeddings)
        #text_embeddings = model.get_text_features(**text_inputs)
        #text_embeddings = F.normalize(text_embeddings, dim=-1)
    return {"image_embeddings": image_embeddings}

    #sim = F.cosine_similarity(text_embeddings, image_embeddings, dim=1)
    #print(sim)


def main():
    if os.path.exists("./data"):
        dset = datasets.load_from_disk("./data")
        print("Loaded dataset from disk")
    else:
        dset = download_dataset()
        dset.save_to_disk("./data")

    print("Dset loaded")
    print(dset)

    dset = dset.select(range(5))
    if os.path.exists("./data_with_embeddings"):
        dset = datasets.load_from_disk("./data_with_embeddings")
    else:
        dset = dset.map(calculate_embeddings)
    print(dset)
    print(dset[4]['image_embeddings'])

if __name__ == "__main__":
    main()
    print("Done totally")