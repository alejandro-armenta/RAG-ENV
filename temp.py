from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer

from langchain.text_splitter import CharacterTextSplitter

url="https://en.wikipedia.org/wiki/2023_Cricket_World_Cup"

loader = AsyncHtmlLoader(url)

html_data = loader.load()

html2text = Html2TextTransformer()

html_data_transformed = html2text.transform_documents(html_data)

#split by character

text_splitter = CharacterTextSplitter(separator='\n', chunk_size=1000, chunk_overlap=200)

#print(type(html_data_transformed[0].page_content))

chunks = text_splitter.create_documents([html_data_transformed[0].page_content])

#es una lista de documentos
#print(type(chunks[0]))
#print(chunks)

print(chunks[4].page_content[-200:])
print()
print(chunks[5].page_content[:200])

