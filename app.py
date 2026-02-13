from flask import Flask, render_template, request
from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from src.prompt import *
from src.helper import *
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)

embedding = download_embeddings()

index_name= 'medical-chatbot'

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embedding
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

llm = ChatGroq(model="llama-3.3-70b-versatile", max_tokens=16384)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt), 
        ("human", "{input}")
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = rag_chain.invoke({"input": msg})
    print("Response: ", response['answer'])
    return str(response['answer'])
 
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)

