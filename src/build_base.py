import os
import io
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import datasets
from datasets import load_dataset
from datasets.utils.file_utils import get_datasets_user_agent

from PIL import Image, ImageFile

USER_AGENT = get_datasets_user_agent()
DATASET_SIZE = 60000
ImageFile.LOAD_TRUNCATED_IMAGES = False


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


def main():
    if os.path.exists("./data"):
        dset = datasets.load_from_disk("./data")
        print("Loaded dataset from disk")
    else:
        dset = download_dataset()
        dset.save_to_disk("./data")

    print("Dset loaded")
    print(dset)

    dset = dset.select(range(1))
    dset[0]['image'].save('img.png')
    print(dset[0]['caption'])


if __name__ == "__main__":
    main()