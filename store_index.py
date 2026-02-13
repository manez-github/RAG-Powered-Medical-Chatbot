from langchain.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
load_dotenv()

from src.helper import *

embedding = download_embeddings()

# pc is the object using which you can access anything and everything that is there on the pinecone website
# pc represents me, the user, going to pinecone and clicking buttons like creating index.
# In technical terms, pc is an instance of a client class that lets your program act as a client and conveniently talk to a remote API/service
pc = Pinecone()

# creating a new vector databse / index
index_name = "medical-chatbot"              # index name = database name

if not pc.has_index(index_name):
    pc.create_index(
        name = index_name,                  
        dimension=384,                      # Should match the dimension of the embedding model
        metric="cosine", 
        spec = ServerlessSpec(cloud="aws", region="us-east-1")
    )
    
index = pc.Index(index_name)

# Actual data which consists of 600+ documents which we extracted from the pdf
extracted_data = load_pdf_files("data")
minimal_docs = filter_to_minimal_docs(extracted_data)
chunks = create_chunks(minimal_docs)

# Inserting data into the vector database
# .from_documents: Converts chunks to vectors and stores it in vector db / index
docsearch = PineconeVectorStore.from_documents(         # docsearch is the vector database 
    documents=chunks,
    embedding=embedding,
    index_name=index_name
)