import os
import shutil

def clear_vectorstore():
    vectorstore_path = "./vectorstore"
    
    if os.path.exists(vectorstore_path):
        shutil.rmtree(vectorstore_path)
        print("Vector store cleared!")
    else:
        print("No vector store found.")

if __name__ == "__main__":
    clear_vectorstore()