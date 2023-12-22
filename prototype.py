# %%
import hashlib
import os
from typing import Iterable

from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from vespa.deployment import VespaDocker
from vespa.package import (
    ApplicationPackage,
    Document,
    Field,
    Schema,
)

# %%
# Define and deploy a Vespa application package using PyVespa.

markdown_schema = Schema(
   name="markdown",
   document=Document(
       fields=[
           Field(name="id", type="string", indexing=["summary", "index"]),
           # array of strings
           Field(name="chunks", type="string[]", indexing=["summary", "index"]),
       ],
   ),
)

# Create the application package
vespa_app_name = "rag"
app_package = ApplicationPackage(name=vespa_app_name, schema=[markdown_schema])

# deploy
vespa_docker = VespaDocker()
vespa_app = vespa_docker.deploy(application_package=app_package)
print("Deployment successful ✨")
# %%
from vespa.io import VespaQueryResponse
import json

response:VespaQueryResponse = vespa_app.query(
    yql="select chunks from markdown where userQuery()",
    groupname="gsajko", 
    query="why is colbert effective?"
)
assert(response.is_successful())
# %%
print(json.dumps(response.hits, indent=2))

# %%
# Utilize LangChain to parse markdown files.
# get all markdown files from folder
# %%
folder_path = "data/md_marker"
md_list = []
for file in os.listdir(folder_path):
    if file.endswith(".md"):
        md_list.append(file)
# %%
# open all files, transform them into Langchain documents
# chunk them
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=1024,
    chunk_overlap=24,
    length_function=len,
    add_start_index=True,
)
docs = []
for path in md_list:
    loader = TextLoader(f"{folder_path}/{path}")
    data = loader.load_and_split(text_splitter=text_splitter)
    text_chunks = [chunk.page_content for chunk in data]
    vespa_id = path
    hash_value = hashlib.sha1(vespa_id.encode()).hexdigest()
    fields = {
        "id": hash_value,
        "chunks": text_chunks,
    }
    docs.append(fields)
# %%

# %%
# Feed the docs to the running Vespa instance.


def vespa_feed(user: str) -> Iterable[dict]:
    for doc in docs:
        yield {"fields": doc, "id": doc["id"], "groupname": user}


from vespa.io import VespaResponse


def callback(response: VespaResponse, id: str):
    if not response.is_successful():
        print(
            f"Document {id} failed to feed with status code {response.status_code}, url={response.url} response={response.json}"
        )


vespa_app.feed_iterable(
    schema="markdown", 
    iter=vespa_feed("gsajko"), namespace="personal", callback=callback
)


