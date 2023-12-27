# %%

import os
from langchain.document_loaders import WebBaseLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from chromadb import PersistentClient

# %%
folder_path = "data/md_marker"
md_list = []
for file in os.listdir(folder_path):
    if file.endswith(".md"):
        md_list.append(file)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = []   
for path in md_list:
    loader = TextLoader(f"{folder_path}/{path}")
    data = loader.load()
    print(f"Loaded {path}")
    splits = text_splitter.split_documents(data)
    print(f"there are {len(splits)} chunks in {path}")
    docs.extend(splits)

# %%
sentence_t_emb = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# %%
vectorstore = Chroma.from_documents(documents=docs, embedding=sentence_t_emb, persist_directory="./chroma_db")
# %%
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=sentence_t_emb)
# vectorstore.add_documents(documents=docs) # it took 15 minutes
# %%
# query it
query = "What is attention?"
retr_docs = vectorstore.similarity_search_with_score(query, k=10)
# The returned distance score is cosine distance. Therefore, a lower score is better.
# %%
# print results
for doc in range(5):
    print(retr_docs[doc][0].page_content)
    print(retr_docs[doc][1])

