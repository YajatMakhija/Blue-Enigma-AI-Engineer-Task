import os
import json
import time
from tqdm import tqdm  # type: ignore
from dotenv import load_dotenv # type: ignore
from pinecone import Pinecone, ServerlessSpec  # type: ignore
import google.generativeai as genai # type: ignore
import config
from log import logger

load_dotenv()

DATA_FILE = "vietnam_travel_dataset.json"
BATCH_SIZE = 32  # Pinecone allows up to 100 vectors per upsert
INDEX_NAME = config.PINECONE_INDEX_NAME
VECTOR_DIM = 1536  # 768 for "models/embedding-001"



class PineconeClient:
    def __init__(self, data = DATA_FILE, index_name = INDEX_NAME, vector_dim = VECTOR_DIM):
        self.data = data
        self.index_name = index_name
        self.vector_dim = vector_dim
        self.pc = Pinecone(api_key=config.PINECONE_API_KEY)
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


    def create_index_and_upload_data(self):
        try:
            existing_indexes = self.pc.list_indexes().names()
            if self.index_name not in existing_indexes:
                print(f"Creating new Pinecone index: {self.index_name}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.vector_dim,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1")
                )
                logger.info(f"[pinecone_upload.py] Index '{self.index_name}' created.")
            else:
                logger.info(f"[pinecone_upload.py] Index '{self.index_name}' already exists.")

            self.index = self.pc.Index(self.index_name)

            with open(self.data, "r", encoding="utf-8") as f:
                nodes = json.load(f)
                items = []
                for node in nodes:
                    text = node.get("semantic_text") or (node.get("description") or "")[:1000]
                    if not text.strip():
                        continue
                    meta = {
                        "id": node.get("id"),
                        "type": node.get("type"),
                        "name": node.get("name"),
                        "city": node.get("city", node.get("region", "")),
                        "tags": node.get("tags", [])
                    }
                    items.append((node["id"], text, meta))

                print(f"\nPreparing to upsert {len(items)} items to Pinecone...\n")

                for batch in tqdm(list(self.chunked(items, BATCH_SIZE)), desc="Uploading batches"):
                    ids = [item[0] for item in batch]
                    texts = [item[1] for item in batch]
                    metas = [item[2] for item in batch]

                    vectors = [
                        {"id": _id, "values": emb, "metadata": meta}
                        for _id, emb, meta in zip(ids, self.get_embeddings(texts), metas)
                    ]

                    self.index.upsert(vectors)
                    time.sleep(0.2)  # small delay to respect rate limits

                logger.info("[pinecone_upload.py] All items uploaded successfully!")
        except Exception as e:
            logger.error(f"[pinecone_upload.py] An error occurred: {e}")
            return


    def get_embeddings(self, texts):
        """Generate vector embeddings for a list of texts."""
        if isinstance(texts, str):
            texts = [texts]
        
        result = genai.embed_content(
            model="gemini-embedding-001",
            content=texts,
            output_dimensionality=self.vector_dim
        )
        
        # Log the embedding generation process
        if result.get("error"):
            logger.error(f"[pinecone_upload.py] Error generating embeddings: {result['error']}")
            return []

        # Handle both single and batch responses
        embeddings = result.get('embedding', [])
        if not isinstance(embeddings[0], list):
            return [embeddings]  # Single text case
        return embeddings


    def chunked(self,iterable, size):
        """Yield successive chunks of size `size` from `iterable`."""
        for i in range(0, len(iterable), size):
            yield iterable[i:i + size]


if __name__ == "__main__":
    client = PineconeClient()
    client.create_index_and_upload_data()