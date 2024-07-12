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

#--------------------------------
from llama_index.core import SimpleDirectoryReader
from llama_index.core import KnowledgeGraphIndex
from llama_index.core import Settings
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.core import StorageContext
from llama_index.llms.huggingface import HuggingFaceInferenceAPI
from langchain.embeddings import HuggingFaceEmbeddings
from llama_index.embeddings.langchain import LangchainEmbedding
from pyvis.network import Network
#PYTHONIOENCODING=utf-8
#llm = Ollama(model="llama3")
import sys
#sys.stdin.reconfigure(encoding='utf-8')
#sys.stdout.reconfigure(encoding='utf-8')
Reader = SimpleDirectoryReader(input_files=["C:\\Users\\DELL\\Desktop\\Photosynthesis_Transcript.pdf"])
documents = Reader.load_data()

#documents.text=documents.text.encode('utf8')
#documents[0].text=documents[0].text.encode('utf8')
for i in range(0,len(documents)):
    documents[i].text = documents[i].text.replace('(', '')
    documents[i].text = documents[i].text.replace(')', '')
    documents[i].text = documents[i].text.replace('- ', ' ')
    documents[i].text = documents[i].text.replace('-', '')
    documents[i].text = documents[i].text.replace('@', 'a')
    documents[i].text = documents[i].text.replace('&', 'and')
    documents[i].text=documents[i].text.encode('utf-8')

print(documents[5].text)

Settings.chunk_size = 1500
Settings.chunk_overlap=100
#from llama_index.core.node_parser import SimpleNodeParser

# Assuming documents have already been loaded

# Initialize the parser
#parser = SimpleNodeParser.from_defaults(chunk_size=1024, chunk_overlap=20)

# Parse documents into nodes
#nodes = parser.get_nodes_from_documents(documents)

import ollama
from llama_index.llms.ollama import Ollama
from llama_index.core import ServiceContext,Settings
#from llama_index.indices.document_summary import DocumentSummaryIndex
from llama_index.core.indices import KeywordTableIndex
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.response_synthesizers import ResponseMode, get_response_synthesizer
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# Embeddings model thenlper/gte-large BAAI/bge-base-en-v1.5
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

# Language model
Settings.llm = Ollama(model="llama3", request_timeout=360.0)


#setup the storage context
graph_store = SimpleGraphStore()
storage_context = StorageContext.from_defaults(graph_store=graph_store)

#Construct the Knowlege Graph Undex
index = KnowledgeGraphIndex.from_documents(documents=documents,
                                           max_triplets_per_chunk=15,
                                           storage_context=storage_context,
                                          include_embeddings=True)

#print(index['263607-263621'])
query = "What are light independent reactions explain in points with detailed reactions."
query_engine = index.as_query_engine(include_text=True,
                                     response_mode ="tree_summarize",
                                     embedding_mode="hybrid",
                                     similarity_top_k=8,)
#
message_template =f"""<|system|>Please check if the following pieces of context has any mention of the  keywords provided in the Question.If not then don't know the answer, just say that you don't know.Stop there.Please donot try to make up an answer.</s>
<|user|>
Question: {query}
Helpful Answer:
</s>"""
#
response = query_engine.query(message_template)
#
print(response.response.split("<|assistant|>")[-1].strip())

#visualize the knowledge graph
from pyvis.network import Network
from IPython.display import display,HTML
g = index.get_networkx_graph()
net = Network(notebook=True,cdn_resources="in_line",directed=True)
net.from_nx(g)
#net.prep_notebook()
#net.show("graph.html")
#net.save_graph("Knowledge_graph.html")

html = net.generate_html()
with open("Knowledge_graph.html", mode='w', encoding='utf-8') as fp:
        fp.write(html)
display(HTML(html))

#
#html = net.generate_html()
#with open("/content/Knowledge_graph.html", mode='w', encoding='utf-8') as fp:
#        fp.write(html)
#display(HTML(html))

#import IPython
#IPython.display.HTML(filename="/content/Knowledge_graph.html")