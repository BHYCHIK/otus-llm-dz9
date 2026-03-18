import chromadb

class VectorStore():
    def __init__(self, path='./chromadb', space='cosine', construction_ef=128, M=64, search_ef=128):
        self._construction_ef = construction_ef
        self._M = M
        self._search_ef = search_ef
        self._space = space

        self._vector_store = chromadb.PersistentClient(path)

        self._collection = self._vector_store.get_or_create_collection('images',
                                                                       metadata={
                                                                           'hnsw:space': self._space,
                                                                           'hnsw:construction_ef': self._construction_ef,
                                                                           'hnsw:M': self._M,
                                                                           'hnsw:search_ef': self._search_ef
                                                                       })

    def save_images(self, embeddings, image_urls):
        self._collection.add(image_urls, embeddings)

    def get_image(self, embedding, top_n=5):
        image = self._collection.query(query_embeddings=embedding,
                                       n_results=top_n)
        return image
