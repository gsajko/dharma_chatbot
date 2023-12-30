# %%
import os

import chromadb
from IPython.display import Markdown, display
from llama_index import ServiceContext, SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.llms import HuggingFaceInferenceAPI
from llama_index.vector_stores import ChromaVectorStore

# %%
folder_path = "data/md_marker"
md_list = []
for file in os.listdir(folder_path):
    if file.endswith(".md"):
        md_list.append(file)


documents = SimpleDirectoryReader(folder_path).load_data()

# %%
#


HF_TOKEN = os.environ["HUGGINGFACEHUB_API_TOKEN"]
repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
remote_llm = HuggingFaceInferenceAPI(model_name=repo_id, token=HF_TOKEN)
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
# %%
# Data Ingestion
# set up ChromaVectorStore and load in data
# vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
# db = chromadb.PersistentClient(path="./chroma_db")
# # chroma_collection = db.get_or_create_collection("quickstart")
# vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
# storage_context = StorageContext.from_defaults(vector_store=vector_store)
# %%
service_context = ServiceContext.from_defaults(embed_model=embed_model, llm=remote_llm)
# %%
# index = VectorStoreIndex.from_documents(
#     documents, storage_context=storage_context, service_context=service_context
# )
# %%
# load from disk
db2 = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db2.get_or_create_collection("quickstart")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
index = VectorStoreIndex.from_vector_store(
    vector_store,
    service_context=service_context,
)
# %%
query_engine = index.as_query_engine()
response = query_engine.query("What is attention?")
display(Markdown(f"<b>{response}</b>"))
# %%
