import faiss

from langchain_community.document_loaders import AsyncHtmlLoader

from langchain_text_splitters import HTMLSectionSplitter

from collections import Counter

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

import matplotlib.pyplot as plt
import numpy as np

url="https://en.wikipedia.org/wiki/2023_Cricket_World_Cup"

loader = AsyncHtmlLoader(url)

html_data = loader.load()

sections_to_split = [
    ('h1','Header 1'),
    ('h2','Header 2'),
    ('table','Table'),
    ('p','Paragraph'),
 ]

splitter = HTMLSectionSplitter(sections_to_split)

split_content = splitter.split_text(html_data[0].page_content)

#print(split_content[:10])

print(len(split_content))

class_counter = Counter()

for doc in split_content:
    document_class = next(iter(doc.metadata.keys()))
    class_counter[document_class] += 1

print(class_counter)

lengths = [len(doc.page_content) for doc in split_content]

plt.figure(1)
plt.boxplot(lengths)

print("Min:",np.min(lengths))
print("25 percentile:",np.percentile(lengths, 25))
print("Median:",np.median(lengths))
print("75 percentile:",np.percentile(lengths, 75))
print("Max:",np.max(lengths))

plt.savefig('lengths.png')

text_splitter = RecursiveCharacterTextSplitter(
    separators=['\n\n','\n','.'],
    chunk_size=1000,
    chunk_overlap=100
    )

final_chunks = text_splitter.split_documents(split_content)

#print(type(final_chunks[0]))

#print(len(final_chunks))

data = [len(doc.page_content) for doc in final_chunks]

plt.figure(2)
plt.boxplot(data)

print("Min:",np.min(data))
print("25 percentile:",np.percentile(data, 25))
print("Median:",np.median(data))
print("75 percentile:",np.percentile(data, 75))
print("Max:",np.max(data))

plt.savefig('fixed_lengths.png')

embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')

hf_embeddings = embeddings.embed_documents([chunk.page_content for chunk in final_chunks])

print(len(final_chunks))

#280*768
print(len(hf_embeddings[0]))

#inner product!
index = faiss.IndexFlatIP(len(hf_embeddings[0]))

vector_store = FAISS(
    embedding_function=embeddings, 
    index=index, 
    docstore=InMemoryDocstore(),
    index_to_docstore_id={}
    )

vector_store.add_documents(documents=final_chunks)

vector_store.save_local(folder_path='./data', index_name='CWC_index')



