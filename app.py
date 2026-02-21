from dotenv import load_dotenv
load_dotenv()

from flask import Flask, render_template, request, session
from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_redis import RedisChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from src.prompt import *
from src.helper import *
import uuid
import os

app = Flask(__name__)

app.secret_key = os.getenv("FLASK_SECRET_KEY")

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
    
    redis_url = os.getenv("REDIS_URL")
    
    if "chat_session_id" not in session:
        session["chat_session_id"] = str(uuid.uuid4())[:12]
    session_id = session["chat_session_id"]
    
    # Gives session history for the respective user(session_id)
    def get_session_history(session_id: str):
        return RedisChatMessageHistory(
            session_id=session_id,
            redis_url=redis_url,
            ttl=3600 * 12
        )
    
    rag_chain_with_memory = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,          
        input_messages_key="input",
        output_messages_key="answer",
        history_messages_key="chat_history"
    )
    
    response = rag_chain_with_memory.invoke(
        {"input": msg}, 
        config={"configurable": {"session_id": session_id}}
        )
    print("Response: ", response['answer'])
    
    print("\n=== Current Chat History ===")
    redis_history = get_session_history(session_id)
    for message in redis_history.messages:
        role = "User" if message.type == "human" else "Bot"
        print(f"{role}: {message.content}")
    print("===========================\n")

    return str(response['answer'])
 
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True)

