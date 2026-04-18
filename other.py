
from langchain_community.vectorstores import FAISS

from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')

vector_store = FAISS.load_local(
    folder_path='./data', 
    embeddings=embeddings, 
    index_name='CWC_index',
    allow_dangerous_deserialization=True
    )

docs = vector_store.similarity_search('Who won the World Cup final?')
#docs = vector_store.similarity_search('Where was the world cup placed?')

print(docs[0].page_content)