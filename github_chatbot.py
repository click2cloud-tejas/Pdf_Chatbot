

import os
from langchain_community.document_loaders import PDFMinerLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import torch

# Function to initialize the QA chain
def initialize_qa_chain(filepaths):
    documents = []
    
    # Load all PDFs
    for filepath in filepaths:
        loader = PDFMinerLoader(filepath)
        documents.extend(loader.load())

    # Split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    splits = text_splitter.split_documents(documents)

    # Create embeddings
    embeddings = SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")

    vectordb = FAISS.from_documents(splits, embeddings)

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
        return_source_documents=True  # Ensures source documents are used for answering
    )
    
    return qa_chain

# Function to process the user's query
def process_answer(instruction, qa_chain):
    # Pass the query as a string and use `invoke` instead of `run`
    result = qa_chain.invoke({"query": instruction})
    
    # Check if any source documents were retrieved
    source_docs = result.get('source_documents', [])
    
    if len(source_docs) == 0:
        return "Sorry, it is not provided in the given context."
    
    # Extract the answer from the result
    answer = result['result']
    
    # Return the generated answer
    return answer

# Main function to handle file uploads and chat interaction in the terminal
def main():
    # Get PDF file paths from the user
    filepaths = []
    num_files = int(input("Enter the number of PDF files you want to upload: "))

    for i in range(num_files):
        filepath = input(f"Enter the full path of PDF file {i+1}: ")
        if os.path.exists(filepath):
            filepaths.append(filepath)
        else:
            print(f"File {filepath} does not exist. Please try again.")
            return

    # Initialize the QA chain
    print("Processing embeddings. This may take some time...")
    qa_chain = initialize_qa_chain(filepaths)
    print("Embeddings processed. You can now ask questions about the PDFs.")

    # Chat loop
    while True:
        prompt = input("\nYou: ")
        if prompt.lower() in ["exit", "quit"]:
            print("Exiting the chatbot.")
            break
        # Pass the prompt (user query) directly as a string
        response = process_answer(prompt, qa_chain)
        print(f"Bot: {response}")

if __name__ == "__main__":
    main()

