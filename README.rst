GraphRetrieval
==============

GraphRetrieval is a Python library designed for advanced text retrieval
and knowledge graph querying. It supports various models and techniques
to enable efficient and accurate information retrieval from large text
corpora and knowledge bases.

Installation
------------

.. code:: bash

   pip install -e git+https://github.com/jayavibhavnk/GraphRetrieval.git#egg=GraphRetrieval

or
--

.. code:: bash

   pip install GraphRetrieval

Usage
~~~~~

Setting Up Environment Variables

Before using the library, set up the necessary environment variables for
Neo4j and OpenAI:

.. code:: python


   import os

   os.environ["NEO4J_URI"] = "add your Neo4j URI here"
   os.environ["NEO4J_USERNAME"] = "add your Neo4j username here"
   os.environ["NEO4J_PASSWORD"] = "add your Neo4j password here"
   os.environ['OPENAI_API_KEY'] = "add your OpenAI API key here"

GraphRAG
--------

GraphRAG is used to create and query graphs based on text documents.

Example
~~~~~~~

.. code:: python


   import GraphRetrieval
   from GraphRetrieval import GraphRAG

   grag = GraphRAG()
   grag.create_graph_from_file('add file path here')

   # Query using the default A* search
   print(grag.queryLLM("Ask your query here")) 

   # Switch to greedy search
   grag.retrieval_model = "greedy"
   print(grag.queryLLM("Ask your query here"))

KnowledgeRAG
------------

KnowledgeRAG integrates with a knowledge graph and supports hybrid
searches combining structured and unstructured data.

.. _example-1:

Example
~~~~~~~

.. code:: python


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

Ingesting Data into Graph
~~~~~~~~~~~~~~~~~~~~~~~~~

Ingest large text data into the knowledge graph.

.. code:: python


   text = """
   some large text here
   """

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

Hybrid Search with GraphRetrieval and Knowledge Base
----------------------------------------------------

Combine GraphRAG and KnowledgeRAG for hybrid search.

.. code:: python


   gr.vector_index = grag
   gr.hybrid = True
   print(gchain.invoke({"question": "Ask your query here"}))

Contributing
^^^^^^^^^^^^

Contributions are welcome! Please submit a pull request or open an issue
to discuss what you would like to change. License

This project is licensed under the MIT License. See the LICENSE file for
details.

This ``README.md`` provides an overview of the GraphRetrieval library,
installation instructions, and example usage scenarios, with the
specified changes to the file path and environment variables sections.
