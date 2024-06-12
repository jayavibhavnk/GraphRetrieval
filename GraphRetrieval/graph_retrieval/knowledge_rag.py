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
from PIL import Image
from torchvision import models, transforms

from ..base.document.py import GraphDocument
from ..utils.text_splitter import CharacterTextSplitter


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

