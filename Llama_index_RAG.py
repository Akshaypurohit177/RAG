#Load the text
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from langchain import hub
#from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
#from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import LanceDB
from langchain_community.embeddings import OllamaEmbeddings

#llm = Ollama(model="llama3")

Reader = SimpleDirectoryReader(input_files=["C:\\Users\\DELL\\Desktop\\Photosynthesis_Transcript.pdf"])
documents = Reader.load_data()

print(documents)


from llama_index.core.node_parser import SimpleNodeParser

# Assuming documents have already been loaded

# Initialize the parser
parser = SimpleNodeParser.from_defaults(chunk_size=1024, chunk_overlap=20)

# Parse documents into nodes
nodes = parser.get_nodes_from_documents(documents)

import ollama
from llama_index.llms.ollama import Ollama
from llama_index.core import ServiceContext,Settings
#from llama_index.indices.document_summary import DocumentSummaryIndex
from llama_index.core.indices import KeywordTableIndex
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.response_synthesizers import ResponseMode, get_response_synthesizer
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# Embeddings model
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

# Language model
Settings.llm = Ollama(model="llama3", request_timeout=360.0)

#llm=Ollama(model="llama3")

#service_context = ServiceContext.from_defaults(llm=Settings.llm,chunk_size=1024)
response_synthesizer = get_response_synthesizer( response_mode="tree_summarize", use_async=True)

doc_summary_index = KeywordTableIndex(nodes)


# Persisting to disk
doc_summary_index.storage_context.persist(persist_dir="C:\\Users\\DELL\\Desktop\\")

# Loading from disk
from llama_index.core import StorageContext, load_index_from_storage
storage_context = StorageContext.from_defaults(persist_dir="C:\\Users\\DELL\\Desktop\\")
index = load_index_from_storage(storage_context)

# Assuming 'index' is your constructed index object
query_engine = index.as_query_engine()
response = query_engine.query("What are light independent reactions explain in points with detailed reactions")
print(response)


