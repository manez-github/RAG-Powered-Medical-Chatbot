from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.schema import Document 
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List

# Converts the pdf pages into a list of documents
def load_pdf_files(path):
    loader = DirectoryLoader(
        path,                    # path = path of the folder in which we have pdfs
        glob="**/*.pdf",         # select all .pdf files from data folder and subfolders in data folder
        loader_cls=PyPDFLoader   # loader_cls = loader class = PyPDF because our book is in pdf format      
    )
    
    documents = loader.load()
    return documents

def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """
    We currently have a list of documents (output of load_pdf_files)
    list = (document_1, document_2, document_3, ....)
    
    Now in each document we have two things
    1. Metadata
    2. Page_Content
    
    Metadata is a dictionary which has various key value pairs 
    But we are only interested in the key 'source'
    
    We will take a list of documents as an input and return a new list of 
    Document Objects containing only 'source' in metadata and page_content
    """
    minimal_docs: List[Document] = []
    for doc in docs: 
        source = doc.metadata.get('source')
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": source}
            )
        )
        
    return minimal_docs


def download_embeddings():
    """ 
    Download and return the HuggingFace embeddings model.
    """
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(
        model_name = model_name
    )
    return embeddings

# Split the document into smaller chunks 
def create_chunks(minimal_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20
    )
    
    chunks = text_splitter.split_documents(minimal_docs)
    return chunks