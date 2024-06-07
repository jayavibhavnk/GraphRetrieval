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



class ImageGraphRAG:
    def __init__(self):
        self.graph = None
        self.documents = None
        self.embeddings = None
        self.embedding_model = models.resnet50(pretrained=True)
        self.embedding_model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.retrieval_model = "a_star"

    def image_to_embedding(self, image):
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        image = self.transform(image)
        image = image.unsqueeze(0)  # add batch dimension
        with torch.no_grad():
            embedding = self.embedding_model(image).numpy().flatten()
        return embedding

    def constructGraph(self, images, similarity_threshold=0.5):
        if all(isinstance(image, str) for image in images):
            documents = [GraphDocument(image, {"path": image}) for image in images]
        else:
            documents = [GraphDocument("Image Object", {"image": image}) for image in images]
        
        embeddings = [self.image_to_embedding(image) for image in images]
        graph = nx.Graph()

        for i in range(len(documents)):
            for j in range(i, len(documents)):
                similarity = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
                if similarity > similarity_threshold:
                    graph.add_edge(i, j, weight=similarity)

        self.graph = graph
        self.documents = documents
        self.embeddings = np.array(embeddings)

        return graph, documents, embeddings

    def a_star_search(self, query_image_path, k=5):
        query_embedding = self.image_to_embedding(query_image_path)
        pq = [(0, None, 0)]
        visited = set()
        similar_nodes = []

        while pq and len(similar_nodes) < k:
            _, current_node, similarity_so_far = heapq.heappop(pq)

            if current_node is not None:
                similar_nodes.append((current_node, similarity_so_far))

            neighbors = self.graph.neighbors(current_node) if current_node is not None else range(len(self.documents))
            for neighbor in neighbors:
                if neighbor not in visited:
                    neighbor_similarity = self.compute_similarity(neighbor, query_embedding)
                    priority = -neighbor_similarity
                    heapq.heappush(pq, (priority, neighbor, similarity_so_far + neighbor_similarity))
                    visited.add(neighbor)

        return similar_nodes

    def compute_similarity(self, neighbor, query_embedding):
        neighbor_embedding = self.embeddings[neighbor]
        neighbor_similarity = cosine_similarity([query_embedding], [neighbor_embedding])[0][0]
        return neighbor_similarity

    def similarity_search(self, query_image_path, retrieval_model="a_star", k=5):
        retrieval_model=self.retrieval_model
        similar_nodes = []

        if retrieval_model == "a_star":
            similar_indices = [index for index, _ in self.a_star_search(query_image_path, k)]

        return [self.documents[index] for index in similar_indices]

    def save_db(self, file_path):
        with open(file_path, 'wb') as file:
            pickle.dump((self.graph, self.documents, self.embeddings), file)
            print("saved!")

    def load_db(self, file_path):
        with open(file_path, 'rb') as file:
            self.graph, self.documents, self.embeddings = pickle.load(file)
            print("loaded!")

    def create_graph_from_directory(self, directory_path=None, images=None, similarity_threshold=0.5):
        if directory_path:
            image_paths = [os.path.join(directory_path, fname) for fname in os.listdir(directory_path) if fname.lower().endswith(('png', 'jpg', 'jpeg'))]
            self.graph, self.documents, self.embeddings = self.constructGraph(image_paths, similarity_threshold)
        elif images:
            self.graph, self.documents, self.embeddings = self.constructGraph(images, similarity_threshold)
        else:
            raise ValueError("Either directory_path or images must be provided.")
        print("Graph created Successfully!")
        return self.documents
        
    def visualize_graph(self):
        pos = nx.spring_layout(self.graph)
        nx.draw(self.graph, pos, with_labels=True, node_color='skyblue', edge_color='gray', node_size=1000, font_size=10)
        plt.show()
