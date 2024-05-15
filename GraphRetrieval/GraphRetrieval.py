from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Tuple
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
import os
from langchain_community.graphs import Neo4jGraph
from langchain.text_splitter import TokenTextSplitter
from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from neo4j import GraphDatabase
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
import langchain_core
import langchain
import os
import networkx as nx
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import heapq
from joblib import Parallel, delayed
from langchain_text_splitters import CharacterTextSplitter
from openai import OpenAI
import pickle
import langchain_core
import PyPDF2
from queue import PriorityQueue


class GraphDocument(langchain_core.documents.base.Document):
    def __init__(self, page_content, metadata):
        super().__init__(page_content=page_content, metadata=metadata)

    def __repr__(self):
        return f"GraphDocument(page_content='{self.page_content}', metadata={self.metadata})"

class Entities(BaseModel):
    """Identifying information about entities."""

    names: List[str] = Field(
        ...,
        description="All the person, organization, or business entities that "
        "appear in the text",
    )

import os
import networkx as nx
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import heapq
from joblib import Parallel, delayed
from langchain_text_splitters import CharacterTextSplitter
from openai import OpenAI
import pickle
import langchain_core
import PyPDF2
from queue import PriorityQueue


class GraphDocument(langchain_core.documents.base.Document):
    def __init__(self, page_content, metadata):
        super().__init__(page_content=page_content, metadata=metadata)

    def __repr__(self):
        return f"GraphDocument(page_content='{self.page_content}', metadata={self.metadata})"
    

class GraphRAG():
    def __init__(self):
        self.graph = None
        self.documents = None
        self.embeddings = None
        self.embedding_model = "all-MiniLM-L6-v2"
        self.retrieval_model = "a_star"

    def constructGraph(self, text, similarity_threshold=0, chunk_size=1250, chunk_overlap=100, metadata=True):
        text_splitter = CharacterTextSplitter(
            separator="\n\n",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

        pre_documents = text_splitter.create_documents([text])
        documents = [GraphDocument(doc.page_content, doc.metadata) for doc in pre_documents]

        model = SentenceTransformer(self.embedding_model)
        embeddings = model.encode([doc.page_content for doc in documents])
        graph = nx.Graph()

        def add_edges(i):
            edges = []
            for j in range(i, len(documents)):
                similarity = cosine_similarity([embeddings[i]], [embeddings[j]])
                if similarity[0][0] > similarity_threshold:
                    edges.append((i, j, similarity[0][0]))
            return edges

        edge_lists = Parallel(n_jobs=-1)(delayed(add_edges)(i) for i in range(len(documents)))
        edges = [edge for edge_list in edge_lists for edge in edge_list]
        graph.add_weighted_edges_from(edges)

        self.graph = graph
        self.documents = documents
        self.embeddings = embeddings

        return graph, documents, embeddings

    def create_graph_from_file(self, file, similarity_threshold=0):
        with open(file, 'r') as file:
            text_data = file.read()
        self.graph, self.documents, self.embeddings = self.constructGraph(text_data, similarity_threshold=similarity_threshold)
        print("Graph created Successfully!")

    def create_graphs_from_directory(self, directory_path, similarity_threshold=0):
        file_list = []
        overall_text = ""
        for file_name in os.listdir(directory_path):
            if file_name.endswith(".txt"):
                file_path = os.path.join(directory_path, file_name)
                temp_text = open(file_path, 'r').read()
                overall_text = overall_text + "\n" + temp_text
                file_list.append((temp_text, file_name))

        self.graph, self.documents, self.embeddings = self.constructGraph(overall_text, similarity_threshold=similarity_threshold)
        print("Graph created Successfully!")

        return file_list

    def create_graph_from_text(self, text, similarity_threshold=0):
        self.graph, self.documents, self.embeddings = self.constructGraph(text, similarity_threshold=similarity_threshold)
        print("Graph created Successfully!")

    def compute_similarity(self, current_node, graph, documents, query_embedding):
        similar_nodes = []
        for neighbor in graph.neighbors(current_node):
            neighbor_embedding = self.embeddings[neighbor]
            neighbor_similarity = cosine_similarity([query_embedding], [neighbor_embedding])[0][0]
            similar_nodes.append((neighbor, neighbor_similarity))
        return similar_nodes

    def a_star_search_parallel(self, graph, documents, embeddings, query_text, k=5):
        model = SentenceTransformer(self.embedding_model)
        query_embedding = model.encode([query_text])[0]

        pq = [(0, None, 0)]
        visited = set()
        similar_nodes = []

        while pq and len(similar_nodes) < k:
            _, current_node, similarity_so_far = heapq.heappop(pq)

            if current_node is not None:
                similar_nodes.append((current_node, similarity_so_far))

            compute_similarity_partial = delayed(self.compute_similarity)
            results = Parallel(n_jobs=-1)(compute_similarity_partial(neighbor, graph, documents, query_embedding) for neighbor in (graph.neighbors(current_node) if current_node is not None else range(len(documents)-1)))

            for result in results:
                for neighbor, neighbor_similarity in result:
                    if neighbor not in visited:
                        priority = -neighbor_similarity
                        heapq.heappush(pq, (priority, neighbor, similarity_so_far + neighbor_similarity))
                        visited.add(neighbor)

        return similar_nodes

    def nearest_neighbors_parallel(self, query_text, k=5):
        model = SentenceTransformer('all-MiniLM-L6-v2')
        query_embedding = model.encode([query_text])[0]
        similarities = cosine_similarity([query_embedding], self.embeddings)
        sorted_indices = sorted(range(len(similarities[0])), key=lambda x: -similarities[0][x])[:k]
        similar_indices = Parallel(n_jobs=-1)(delayed(self._get_similarity)(i, similarities, sorted_indices) for i in range(k))
        return similar_indices

    def _get_similarity(self, i, similarities, sorted_indices):
        idx = sorted_indices[i]
        return idx, similarities[0][idx]

    def nearest_neighbors_sklearn_parallel(self, embeddings, query_text, k=5):
        model = SentenceTransformer('all-MiniLM-L6-v2')
        query_embedding = model.encode([query_text])[0]

        nn_model = NearestNeighbors(n_neighbors=k, metric='cosine')
        nn_model.fit(embeddings)

        distances, indices = nn_model.kneighbors([query_embedding])

        similar_indices = Parallel(n_jobs=-1)(delayed(self._get_similarity_sklearn)(i, distances, indices) for i in range(k))
        return similar_indices

    def _get_similarity_sklearn(self, i, distances, indices):
        idx = indices[0][i]
        return idx, 1 - distances[0][i]

    def greedy_bfs_search_parallel(self, query_text, k=5):
        model = SentenceTransformer(self.embedding_model)
        query_embedding = model.encode([query_text])[0]

        pq = PriorityQueue()
        pq.put((0, None, 0))
        visited = set()
        similar_nodes = []

        while not pq.empty() and len(similar_nodes) < k:
            _, current_node, similarity_so_far = pq.get()

            if current_node is not None:
                similar_nodes.append((current_node, similarity_so_far))

            compute_similarity_partial = delayed(self.compute_similarity)
            results = Parallel(n_jobs=-1)(compute_similarity_partial(neighbor, self.graph, self.documents, query_embedding) for neighbor in (self.graph.neighbors(current_node) if current_node is not None else range(len(self.documents)-1)))

            for result in results:
                for neighbor, neighbor_similarity in result:
                    if neighbor not in visited:
                        priority = -neighbor_similarity
                        pq.put((priority, neighbor, similarity_so_far + neighbor_similarity))
                        visited.add(neighbor)

        return similar_nodes

    def similarity_search(self, query, retrieval_model="a_star", k = 5):
        retrieval_model=self.retrieval_model
        similar_nodes = []

        if retrieval_model == "a_star":
            similar_indices = [index for index, _ in self.a_star_search_parallel(self.graph, self.documents, self.embeddings, query, k)]
        elif retrieval_model == "nearest_neighbors":
            similar_indices = [index for index, _ in self.nearest_neighbors_parallel(query, k)]
        elif retrieval_model == "nearest_neighbors1":
            similar_indices = [index for index, _ in self.nearest_neighbors_sklearn_parallel(self.embeddings, query, k)]
        elif retrieval_model == "greedy":
            similar_indices = [index for index, _ in self.greedy_bfs_search_parallel(query, k)]

        return [self.documents[index] for index in similar_indices]

    def queryLLM(self, query):
        similar_documents = self.similarity_search(query)
        full_text = "\n".join([doc.page_content for doc in similar_documents])

        prompt_template = """
        You are an assistant that answers user's queries, you will be given a context and and some instruction, you will answer the query based on this,
        context: {context},
        query: {query}
        """
        ans = self.query_openai(prompt_template.format(context=full_text, query=query))

        return ans

    def query_openai(self, query):
        client = OpenAI()
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": query}
            ],
            n=1
        )
        return(completion.choices[0].message.content)

    def save_db(self, file_path):
        with open(file_path, 'wb') as file:
            pickle.dump((self.graph, self.documents, self.embeddings), file)
            print("saved!")

    def load_db(self, file_path):
        with open(file_path, 'rb') as file:
            self.graph, self.documents, self.embeddings = pickle.load(file)
            print("loaded!")

    def update_graph_with_new_text(self, new_text, similarity_threshold=0):
        new_documents = [GraphDocument(text, None) for text in new_text.split('\n\n')]
        model = SentenceTransformer(self.embedding_model)
        new_embeddings = model.encode([doc.page_content for doc in new_documents])

        if len(self.documents) > 0 and len(new_documents) > 0:
            similarity = cosine_similarity(self.embeddings, new_embeddings)
            edges = [(i, j + len(self.documents), similarity[i][j]) for i in range(len(self.documents)) for j in range(len(new_documents))
                    if similarity[i][j] > similarity_threshold]
            self.graph.add_weighted_edges_from(edges)

        def add_edges(i):
            edges = []
            for j in range(i, len(new_documents)):
                similarity = cosine_similarity([self.embeddings[i]], [new_embeddings[j]])
                if similarity[0][0] > similarity_threshold:
                    edges.append((i, j + len(self.documents), similarity[0][0]))
            return edges

        edge_lists = Parallel(n_jobs=-1)(delayed(add_edges)(i) for i in range(len(self.documents)))
        edges = [edge for edge_list in edge_lists for edge in edge_list]
        self.graph.add_weighted_edges_from(edges)

        self.documents.extend(new_documents)
        self.embeddings.extend(new_embeddings)

        return self.graph, self.documents, self.embeddings

    def create_graph_from_pdf(self, pdf_file, similarity_threshold=0):
        with open(pdf_file, "rb") as f:
            pdf_reader = PyPDF2.PdfFileReader(f)
            num_pages = pdf_reader.numPages
            text = ""
            for page_num in range(num_pages):
                text += pdf_reader.getPage(page_num).extractText()
        self.graph, self.documents, self.embeddings = self.constructGraph(text, similarity_threshold=similarity_threshold)
        print("Graph created Successfully!")   



_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question,
in its original language.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""  # noqa: E501
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

def _format_chat_history(chat_history: List[Tuple[str, str]]) -> List:
    buffer = []
    for human, ai in chat_history:
        buffer.append(HumanMessage(content=human))
        buffer.append(AIMessage(content=ai))
    return buffer

_search_query = RunnableBranch(
    # If input includes chat_history, we condense it with the follow-up question
    (
        RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
            run_name="HasChatHistoryCheck"
        ),  # Condense follow-up question and chat into a standalone_question
        RunnablePassthrough.assign(
            chat_history=lambda x: _format_chat_history(x["chat_history"])
        )
        | CONDENSE_QUESTION_PROMPT
        | ChatOpenAI(temperature=0)
        | StrOutputParser(),
    ),
    # Else, we have no chat history, so just pass through the question
    RunnableLambda(lambda x : x["question"]),
)


class KnowledgeRAG():
    def __init__(self):
        self.hybrid = False
        self.graph = None
        self.llm = self.LLMchain()
        self.entity_chain = self.create_entity_chain()
        self._template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question,
        in its original language.
        Chat History:
        {chat_history}
        Follow Up Input: {question}
        Standalone question:"""  # noqa: E501
        self.CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(self._template)
        self.vector_index = None

    def LLMchain(self, **vars):
        llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-0125")
        return llm

    def load_text_into_graph(self, text, chunk_size = 1500, chunk_overlap = 150):
        text_splitter = CharacterTextSplitter(
            separator="\n\n",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        docs1 = text_splitter.create_documents([file])
        docs = gr.generate_graph_from_text(docs1)
        self.ingest_data_into_graph(docs)
    
    def create_entity_chain(self):
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are extracting organization and person entities from the text.",
                ),
                (
                    "human",
                    "Use the given format to extract information from the following "
                    "input: {question}",
                ),
            ]
        )

        entity_chain = prompt | self.llm.with_structured_output(Entities)

        return entity_chain

    def init_graph(self, graph = None, **neo4jvars):
        if graph == None:
            self.graph = Neo4jGraph(**neo4jvars)
        else:
            self.graph = graph

    def ingest_data_into_graph(self, graph_documents):
        self.graph.add_graph_documents(
        graph_documents,
        baseEntityLabel=True,
        include_source=True
        )
        self.graph.query("CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id]")

        print("ingested!")

    
    def generate_graph_from_text(self, documents):
        if self.llm != None:
            llm_transformer = LLMGraphTransformer(llm=self.llm)
        else:
            llm_transformer = LLMGraphTransformer(llm=self.llm)
            self.llm = self.llm

        graph_documents = llm_transformer.convert_to_graph_documents(documents)

        return graph_documents
    
    def init_neo4j_vector_index(self):
        vector_index = Neo4jVector.from_existing_graph(
            OpenAIEmbeddings(),
            search_type="hybrid",
            node_label="Document",
            text_node_properties=["text"],
            embedding_node_property="embedding"
        )

        self.vector_index = vector_index
    
    def generate_full_text_query(self, input: str) -> str:
        """
        Generate a full-text search query for a given input string.

        This function constructs a query string suitable for a full-text search.
        It processes the input string by splitting it into words and appending a
        similarity threshold (~2 changed characters) to each word, then combines
        them using the AND operator. Useful for mapping entities from user questions
        to database values, and allows for some misspellings.
        """
        full_text_query = ""
        words = [el for el in remove_lucene_chars(input).split() if el]
        for word in words[:-1]:
            full_text_query += f" {word}~2 AND"
        full_text_query += f" {words[-1]}~2"
        return full_text_query.strip()

    def structured_retriever(self, question: str) -> str:
        """
        Collects the neighborhood of entities mentioned
        in the question
        """
        result = ""
        entities = self.entity_chain.invoke({"question": question})
        for entity in entities.names:
            response = self.graph.query(
                """CALL db.index.fulltext.queryNodes('entity', $query, {limit:2})
                YIELD node,score
                CALL {
                WITH node
                MATCH (node)-[r:!MENTIONS]->(neighbor)
                RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
                UNION ALL
                WITH node
                MATCH (node)<-[r:!MENTIONS]-(neighbor)
                RETURN neighbor.id + ' - ' + type(r) + ' -> ' +  node.id AS output
                }
                RETURN output LIMIT 50
                """,
                {"query": self.generate_full_text_query(entity)},
            )
            result += "\n".join([el['output'] for el in response])
        return result
    
    def retriever(self, question: str):
        # print(f"Search query: {question}")
        structured_data = self.structured_retriever(question)
        if self.vector_index != None and self.hybrid==True:
            unstructured_data = [el.page_content for el in self.vector_index.similarity_search(question)] 
        else:
            unstructured_data = ""
        final_data = f"""Structured data:
            {structured_data}
            Unstructured data:
            {"#Document ". join(unstructured_data)}
        """
        return final_data
    
    def graphChain(self):
        template = """Answer the question based only on the following context:
        {context}

        Question: {question}
        Use natural language and be concise.
        Answer:"""
        prompt = ChatPromptTemplate.from_template(template)

        chain = (
            RunnableParallel(
                {
                    "context": _search_query | self.retriever,
                    "question": RunnablePassthrough(),
                }
            )
            | prompt
            | self.llm
            | StrOutputParser()
        )

        return chain
