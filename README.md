This code runs only with the pdf with OCR
1) Model = all-mpnet-base-v2 => to create embeddings and also to retrieve them
2) FAISS (Facebook AI Similarity Search) => VectorDB (to store the embeddings)
3) Model= Lamini-T5( developed by UAE University)=> to refine the response and to store the conversation
