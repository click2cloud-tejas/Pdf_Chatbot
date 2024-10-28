import os
import requests
from bs4 import BeautifulSoup
from langchain_huggingface import HuggingFaceEmbeddings as SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from langchain.schema import Document
from urllib.parse import urljoin
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import torch



# Define FAISS vector database directory
FAISS_DB_DIR = 'faiss_index2'

# Function to scrape text content from a URL
def scrape_website(url):
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to retrieve {url}")
        return ""
    
    soup = BeautifulSoup(response.content, 'html.parser')
    # Extract text content; customize tags based on website structure
    text = ' '.join([p.get_text() for p in soup.find_all(['p', 'h1', 'h2', 'h3'])])
    return text

# Function to retrieve all internal links on a webpage
def get_absolute_links(url):
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to retrieve links from {url}")
        return []
    
    soup = BeautifulSoup(response.text, 'html.parser')
    # Convert all found links to absolute URLs and filter out unwanted formats
    links = [urljoin(url, a.get('href')) for a in soup.find_all('a', href=True)]
    links = [link for link in links if not link.endswith(('.jpg', '.jpeg', '.png', '.gif'))]
    return list(set(links))  # Remove duplicate links

# Function to initialize the QA chain with an option to load or create a FAISS vectorDB
def initialize_qa_chain(urls):
    # Ask user whether to load existing vectorDB or create new embeddings
    if os.path.exists(FAISS_DB_DIR) and input("Load existing FAISS vectorDB? (y/n): ").strip().lower() == 'y':
        print("Loading existing FAISS vectorDB...")
        
        vectordb = FAISS.load_local(FAISS_DB_DIR, SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2"), allow_dangerous_deserialization=True)

    else:
        print("Creating new embeddings from provided URLs...")
        documents = []

        # Scrape and load all website contents, including internal links
        for url in urls:
            website_text = scrape_website(url)
            if website_text:
                documents.append(Document(page_content=website_text, metadata={"source": url}))

            internal_links = get_absolute_links(url)
            for link in internal_links:
                internal_text = scrape_website(link)
                if internal_text:
                    documents.append(Document(page_content=internal_text, metadata={"source": link}))

        # Split the documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)
        splits = text_splitter.split_documents(documents)

        # Create embeddings
        embeddings = SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")
        vectordb = FAISS.from_documents(splits, embeddings)

        # Save the vectorDB to the specified path
        vectordb.save_local(FAISS_DB_DIR)
        print(f"FAISS vectorDB saved at {FAISS_DB_DIR}.")

    # Initialize the LaMini-T5 model
    CHECKPOINT = "MBZUAI/LaMini-T5-738M"
    TOKENIZER = AutoTokenizer.from_pretrained(CHECKPOINT)
    BASE_MODEL = AutoModelForSeq2SeqLM.from_pretrained(CHECKPOINT, device_map=torch.device('cpu'), torch_dtype=torch.float32)
    pipe = pipeline(
        'text2text-generation',
        model=BASE_MODEL,
        tokenizer=TOKENIZER,
        max_length=1024,
        do_sample=True,
        temperature=0.3,
        top_p=0.95,
    )
    llm = HuggingFacePipeline(pipeline=pipe)

    # Build a QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever(),
        return_source_documents=True
    )

    return qa_chain

# Function to process the user's query
def process_answer(instruction, qa_chain):
    result = qa_chain.invoke({"query": instruction})
    source_docs = result.get('source_documents', [])

    if len(source_docs) == 0:
        return "Sorry, it is not provided in the given context."
    
    # Retrieve answer and source URL from the result
    answer = result['result']
    source_url = source_docs[0].metadata.get("source", "Unknown source")
    
    return f"{answer}\n\nSource: {source_url}"

# Function to read URLs from a file
def get_urls_from_file(filename):
    if not os.path.exists(filename):
        print(f"{filename} not found.")
        return []
    
    with open(filename, 'r') as file:
        urls = [line.strip() for line in file if line.strip()]
    
    return urls

# Main function to handle chat interaction
def main():
    #filename = 'agriculture_links.txt'  # Change filename as needed
    filename = 'test_link.txt'

    urls = get_urls_from_file(filename)

    if not urls:
        print(f"No URLs found in {filename}. Exiting.")
        return

    # Initialize the QA chain with option to load or create FAISS vectorDB
    print("Processing embeddings. This may take some time...")
    qa_chain = initialize_qa_chain(urls)
    print("Embeddings processed. You can now ask questions about the URLs.")

    # Chat loop
    while True:
        prompt = input("\nYou: ")
        if prompt.lower() in ["exit", "quit", "bye"]:
            print("Exiting the chatbot.")
            break
        response = process_answer(prompt, qa_chain)
        print(f"Bot: {response}")

if __name__ == "__main__":
    main()


