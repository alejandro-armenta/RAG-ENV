from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer

url="https://en.wikipedia.org/wiki/2023_Cricket_World_Cup"

loader = AsyncHtmlLoader(url)

#print(loader)

html_data = loader.load()

#print(html_data)


html2text = Html2TextTransformer()

html_data_transformed = html2text.transform_documents(html_data)

print(html_data_transformed[0].page_content)

#build custom connectors and loader!

#Lost in the middle problem!

