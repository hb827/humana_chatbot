import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def create_knowledge_base(pdf_location):
    pdf_path=pdf_location+"/slamon_etal.pdf"
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            print("Found a page! Extracting text...")
            left = page.crop((0, 0, 0.5 * page.width, 0.9 * page.height))
            right = page.crop((0.5 * page.width, 0, page.width, page.height))
            l_text = left.extract_text(x_tolerance=1)
            r_text = right.extract_text(x_tolerance=1)
            page_text = l_text + " " + r_text
            if page_text:
                text += page_text

    # save full text of pdf for synthetic benchmarking set creation
    with open('parsed_documents/slamon_etal.txt', 'w') as f: 
        f.write(text) 

    splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100)
    chunks = splitter.split_text(text)

    print("Getting embeddings...")
    model_choice = 'sentence-transformers/all-MiniLM-L6-v2'
    model = HuggingFaceEmbeddings(model_name=model_choice) 

    print("Creating FAISS index...") 
    vectorstore = FAISS.from_texts(chunks, model)

    # Save everything 
    print("Saving knowledge base and index...") 
    index_name = "knowledge_base"
    vectorstore.save_local(folder_path="", index_name=index_name)
    
    # Save model name for later loading 
    with open('model_name.txt', 'w') as f: 
        f.write(model_choice) 
    

if __name__ == "__main__":
    create_knowledge_base("documents")
