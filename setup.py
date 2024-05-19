from setuptools import setup, find_packages

long_description = """
# GraphRetrieval

GraphRetrieval is a Python library designed for advanced text retrieval and knowledge graph querying. It supports various models and techniques to enable efficient and accurate information retrieval from large text corpora and knowledge bases.

## Installation

```bash
pip install -e git+https://github.com/jayavibhavnk/GraphRetrieval.git#egg=GraphRetrieval
```

or 

```bash
pip install GraphRetrieval
```

Usage
Setting Up Environment Variables

Before using the library, set up the necessary environment variables for Neo4j and OpenAI:

```python

import os

os.environ["NEO4J_URI"] = "add your Neo4j URI here"
os.environ["NEO4J_USERNAME"] = "add your Neo4j username here"
os.environ["NEO4J_PASSWORD"] = "add your Neo4j password here"
os.environ['OPENAI_API_KEY'] = "add your OpenAI API key here"
```

## GraphRAG

GraphRAG is used to create and query graphs based on text documents.

### Example

```python

import GraphRetrieval
from GraphRetrieval import GraphRAG

grag = GraphRAG()
grag.create_graph_from_file('add file path here')

# Query using the default A* search
print(grag.queryLLM("Ask your query here")) 

# Switch to greedy search
grag.retrieval_model = "greedy"
print(grag.queryLLM("Ask your query here"))
```

## KnowledgeRAG

KnowledgeRAG integrates with a knowledge graph and supports hybrid searches combining structured and unstructured data.

### Example

```python

from GraphRetrieval import KnowledgeRAG
from langchain_community.graphs import Neo4jGraph

graph = Neo4jGraph()
gr = KnowledgeRAG()

# Initialize graph
gr.init_graph(graph)

# Create the graph chain
gchain = gr.graphChain()

# Query the graph chain
print(gchain.invoke({"question": "Ask your query here"}))

# Hybrid search using Neo4j vector index
gr.init_neo4j_vector_index()
gr.hybrid = True
print(gchain.invoke({"question": "Ask your query here"}))
```

### Ingesting Data into Graph

Ingest large text data into the knowledge graph.

```python

text = "enter text here"

from langchain_text_splitters import CharacterTextSplitter

text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)

docs1 = text_splitter.create_documents([text])
docs = gr.generate_graph_from_text(docs1)
gr.ingest_data_into_graph(docs)

gr.init_neo4j_vector_index()
print(gchain.invoke({"question": "Ask your query here"}))
```

## Hybrid Search with GraphRetrieval and Knowledge Base

Combine GraphRAG and KnowledgeRAG for hybrid search.

```python

gr.vector_index = grag
gr.hybrid = True
print(gchain.invoke({"question": "Ask your query here"}))
```

## Image Graph RAG

Use directories of images for searching similar images.

```python
image_graph_rag = ImageGraphRAG()
image_paths = image_graph_rag.create_graph_from_directory('/content/images')
similar_images = image_graph_rag.similarity_search('/content/images/car.jpg', k=5)

for doc in similar_images:
    print(doc.metadata["path"])
```

```python
image_graph_rag.visualize_graph() # for graph visualization
```



#### Contributing

Contributions are welcome! Please submit a pull request or open an issue to discuss what you would like to change.
License

This project is licensed under the MIT License. See the LICENSE file for details.

This `README.md` provides an overview of the GraphRetrieval library, installation instructions, and example usage scenarios, with the specified changes to the file path and environment variables sections.
"""

setup(
    name='GraphRetrieval',
    version='0.1.3',
    description='Graph retrieval',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='JVNK',
    author_email='jaya11vibhav@gmail.com',
    url='https://github.com/jayavibhavnk/GraphRetrieval',
    packages=find_packages(),
    install_requires=[
      'langchain_openai',
      'langchain',
      'sentence_transformers',
      'langchain-community', 
      'langchain-openai',
      'langchain-experimental',
      'neo4j',
      'wikipedia',
      'tiktoken',
      'pypdf2'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha'
    ],
)
