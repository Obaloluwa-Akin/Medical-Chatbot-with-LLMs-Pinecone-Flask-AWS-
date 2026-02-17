from dotenv import load_dotenv
import os
import time
from src.helper import load_pdf_file, filter_to_minimal_docs, text_split, download_hugging_face_embeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

if not PINECONE_API_KEY or not OPENAI_API_KEY:
    raise ValueError("Missing API keys in .env file")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

print("Loading PDF files...")
extracted_data = load_pdf_file(data='data/')
print(f"Loaded {len(extracted_data)} documents")

print("Filtering documents...")
filter_data = filter_to_minimal_docs(extracted_data)

print("Splitting text into chunks...")
text_chunks = text_split(filter_data)
print(f"Created {len(text_chunks)} text chunks")

print("Loading embeddings model...")
embeddings = download_hugging_face_embeddings()

print("Connecting to Pinecone...")
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medical-chatbot"

# Check if index exists using list_indexes()
existing_indexes = pc.list_indexes()
index_names = [index['name'] for index in existing_indexes]

if index_name not in index_names:
    print(f"Creating index '{index_name}'...")
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    print(f"Index '{index_name}' created successfully!")
    
    # Wait for index to be ready
    print("Waiting for index to be ready...")
    time.sleep(60)  # Wait 60 seconds for the index to initialize
    
    # Check index status
    while not pc.describe_index(index_name).status['ready']:
        print("Index still initializing... waiting 10 more seconds")
        time.sleep(10)
    
    print("Index is ready!")
else:
    print(f"Index '{index_name}' already exists.")

print("Uploading documents to Pinecone (this may take several minutes)...")

try:
    # Upload in smaller batches to avoid connection issues
    batch_size = 100
    for i in range(0, len(text_chunks), batch_size):
        batch = text_chunks[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(text_chunks)-1)//batch_size + 1} ({len(batch)} chunks)...")
        
        if i == 0:
            # Create the vector store with the first batch
            docsearch = PineconeVectorStore.from_documents(
                documents=batch,
                index_name=index_name,
                embedding=embeddings,
            )
        else:
            # Add subsequent batches
            docsearch.add_documents(batch)
        
        time.sleep(2)  # Small delay between batches
        
    print("✅ All documents successfully uploaded to Pinecone!")
    print(f"Total chunks indexed: {len(text_chunks)}")
    
except Exception as e:
    print(f"❌ Error uploading documents: {str(e)}")
    print("The index has been created. You can try running this script again.")
    raise