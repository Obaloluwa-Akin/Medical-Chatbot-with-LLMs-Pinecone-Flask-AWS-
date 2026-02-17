from flask import Flask, render_template, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from src.prompt import system_prompt
import os

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

if not PINECONE_API_KEY or not OPENAI_API_KEY:
    raise ValueError("Missing API keys in .env file")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

print("Loading embeddings model...")
embeddings = download_hugging_face_embeddings()

index_name = "medical-chatbot"
print(f"Connecting to Pinecone index: {index_name}")
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Build the RAG chain
rag_chain = (
    RunnableParallel({
        "context": retriever | format_docs,
        "input": RunnablePassthrough()
    })
    | prompt
    | llm
    | StrOutputParser()
)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    print(f"Input: {msg}")
    
    try:
        response = rag_chain.invoke(msg)
        print(f"Response: {response}")
        return str(response)
    except Exception as e:
        print(f"Error: {str(e)}")
        return f"Sorry, I encountered an error: {str(e)}"

if __name__ == '__main__':
    print("✅ Flask app is ready!")
    print("🌐 Access the chatbot at: http://127.0.0.1:8080")
    app.run(host="0.0.0.0", port=8080, debug=True)