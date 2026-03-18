from fastapi import FastAPI, Query
from .rag.image_index import VectorStore
from .rag.embedder import Embedder

app = FastAPI()

vector_store = VectorStore()
embedder = Embedder()
@app.get("/")
def root():
    return {"status": "ok"}

@app.get("/find_images")
def find_images(
        image_description: str = Query(..., description="Description of image"),
        top_n: int = Query(5, ge=1, le=50)
                ):

    text_embeddings = embedder.embed_text(image_description)
    res = vector_store.get_image(text_embeddings, top_n=top_n)
    print(res)

    urls = res['ids'][0]

    return {
        "description": image_description,
        "top_n": top_n,
        "urls": urls
    }