# First RAG Application using LLma 3 and Langchain
from langchain import hub
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
#from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import LanceDB
from langchain_community.embeddings import OllamaEmbeddings
from chromadb import Embeddings
# Create an embedding with the correct dimensionality
#embedder = Embeddings.embedder(dimension=5120)
from langchain.document_loaders import PyPDFLoader

#Load the text
llm = Ollama(model="llama3")
loader = DirectoryLoader("C:\\Users\\DELL\\Desktop", glob="Photosynthesis_Transcript.pdf")
books = loader.load()
#print("no: of pages-  ",len(pdf_pages))
#print(pages[0].page_content)
# Split the text

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
all_splits = text_splitter.split_documents(books)

#embeddings & chroma vector store
from lancedb.rerankers import LinearCombinationReranker

reranker = LinearCombinationReranker(weight=0.5)
vectorstore_result = LanceDB.from_documents(documents=all_splits,
                                            embedding=OllamaEmbeddings(model='llama3', show_progress=True),
                                            persist_directory=".//LanceDB", )


#db = LanceDB.connect(".//LanceDB")

#testing the similarity search
question = "what are Light Dependent Reactions"
docs = vectorstore_result.similarity_search(question)
print("text - ", docs[0])
tbl = vectorstore_result.get_table()

#LanceDB.write_dataset(tbl, ".//plant.lance")
print("tbl:", tbl)
pd_df = tbl.to_pandas()
pd_df.to_csv("C:\\Users\\DELL\\Desktop\\docsearch.csv", index=False)
#print(pd_df)

# now pass similarity search result to model along with question
#vectorstore = LanceDB.connect()
retriever = vectorstore_result.as_retriever()

rag_prompt = hub.pull("rlm/rag-prompt")
qa_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
)

question = "What are light independent processes explain in points with detailed reactions?"
answer=qa_chain.invoke(question)

print("answer-",answer)