from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from .rag.image_index import VectorStore
from .rag.embedder import Embedder
from .website.html import get_main_page_html, get_serp_page_html
app = FastAPI()

vector_store = VectorStore()
embedder = Embedder()
@app.get("/", response_class=HTMLResponse)
def root():
    return get_main_page_html()

@app.get("/find_images", response_class=HTMLResponse)
def find_images(
        image_description: str = Query(..., description="Description of image"),
        top_n: int = Query(5, ge=1, le=50)
                ):

    text_embeddings = embedder.embed_text(image_description)
    res = vector_store.get_image(text_embeddings, top_n=top_n)
    print(res)

    urls = res['ids'][0]

    return get_serp_page_html(urls)