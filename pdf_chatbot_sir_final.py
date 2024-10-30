import os
from langchain_community.document_loaders import PDFMinerLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from langchain.schema import Document
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import torch
import tabula
import pandas as pd
from pdf2image import convert_from_path
import pytesseract

# Path where FAISS index will be saved locally
FAISS_INDEX_PATH = "faiss_index_final_sir"

# Function to convert table rows into meaningful sentences
def convert_table_to_sentences(table: pd.DataFrame) -> list:
    sentences = []
    for index, row in table.iterrows():
        row_sentence = ". ".join([f"{col}: {row[col]}" for col in table.columns])
        sentences.append(row_sentence)
    return sentences

# Function to extract tables from PDF using Tabula
def extract_tables_from_pdf(filepath):
    tables = tabula.read_pdf(filepath, pages="all", multiple_tables=True)
    return tables

# Function to perform OCR on scanned PDF pages
def ocr_pdf(filepath):
    text = ""
    pages = convert_from_path(filepath, dpi=300)
    for page in pages:
        page_text = pytesseract.image_to_string(page)
        text += page_text
    return text

# Function to load PDF with OCR fallback for scanned PDFs
def load_pdf_with_ocr(filepath):
    try:
        loader = PDFMinerLoader(filepath)
        doc_texts = loader.load()
        
        if not doc_texts or all(not doc.page_content.strip() for doc in doc_texts):
            print("Scanned PDF detected; performing OCR...")
            doc_texts = [Document(page_content=ocr_pdf(filepath))]
        
        return doc_texts
    except Exception as e:
        print("Error loading PDF:", e)
        return []

# Function to initialize the QA chain
def initialize_qa_chain(vectordb):
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
    answer = result['result']
    return answer

# Function to load or create a vector database
def load_or_create_vectordb(embeddings):
    if os.path.exists(FAISS_INDEX_PATH):
        choice = input("Do you want to load the existing vector database or add more PDFs? (load/add): ").lower()
        
        if choice == "load":
            print("Loading existing FAISS index...")
            vectordb = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
            return vectordb
        
        elif choice == "add":
            vectordb = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
            vectordb = add_new_pdfs_to_vectordb(vectordb, embeddings)
            return vectordb
        else:
            print("Invalid option, loading existing FAISS index by default.")
            vectordb = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
            return vectordb
    else:
        print("No existing FAISS index found. You must upload PDFs.")
        return create_new_vectordb(embeddings)

# Function to add new PDFs and update vector database
def add_new_pdfs_to_vectordb(vectordb, embeddings):
    filepaths = []
    num_files = int(input("Enter the number of PDF files you want to add: "))
    
    for i in range(num_files):
        filepath = input(f"Enter the full path of PDF file {i+1}: ")
        if os.path.exists(filepath):
            filepaths.append(filepath)
        else:
            print(f"File {filepath} does not exist. Please try again.")
            return vectordb
    
    documents = []
    for filepath in filepaths:
        doc_texts = load_pdf_with_ocr(filepath)
        documents.extend(doc_texts)

        # Extract tables from the PDF
        tables = extract_tables_from_pdf(filepath)
        for table in tables:
            table_sentences = convert_table_to_sentences(table)
            documents.extend([Document(page_content=sentence) for sentence in table_sentences])
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)
    splits = text_splitter.split_documents(documents)
    
    vectordb.add_documents(splits)
    vectordb.save_local(FAISS_INDEX_PATH)
    print(f"New PDFs and tables added, and FAISS index updated. Saved to {FAISS_INDEX_PATH}.")
    
    return vectordb

# Function to create a new vector database if it doesn't exist
def create_new_vectordb(embeddings):
    filepaths = []
    num_files = int(input("Enter the number of PDF files you want to upload: "))

    for i in range(num_files):
        filepath = input(f"Enter the full path of PDF file {i+1}: ")
        if os.path.exists(filepath):
            filepaths.append(filepath)
        else:
            print(f"File {filepath} does not exist. Please try again.")
            return None

    documents = []
    for filepath in filepaths:
        doc_texts = load_pdf_with_ocr(filepath)
        documents.extend(doc_texts)

        # Extract tables from the PDF
        tables = extract_tables_from_pdf(filepath)
        for table in tables:
            table_sentences = convert_table_to_sentences(table)
            documents.extend([Document(page_content=sentence) for sentence in table_sentences])

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)
    splits = text_splitter.split_documents(documents)

    vectordb = FAISS.from_documents(splits, embeddings)
    vectordb.save_local(FAISS_INDEX_PATH)
    print(f"New FAISS index created and saved to {FAISS_INDEX_PATH}.")
    
    return vectordb

# Main function to handle file uploads and chat interaction in the terminal
def main():
    embeddings = SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")
    vectordb = load_or_create_vectordb(embeddings)
    qa_chain = initialize_qa_chain(vectordb)
    print("Embeddings are ready. You can now ask questions about the PDFs.")

    while True:
        prompt = input("\nYou: ")
        if prompt.lower() in ["exit", "quit"]:
            print("Exiting the chatbot.")
            break
        response = process_answer(prompt, qa_chain)
        print(f"Bot: {response}")

if __name__ == "__main__":
    main()


