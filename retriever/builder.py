#====Changes===
# 1. WatsonxEmbeddings and all related watson packages.



from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames
#from langchain_ibm import WatsonxEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from config.settings import settings
import logging

logger = logging.getLogger(__name__)

#Purpose: Implements hybrid retrieval system combining 1) BM25 (Lexical Search - keyword-based retrieval) and 2) Vector Search (Embedding-search; semantic retrieval using embeddings)
#Why? Captures both exact keyword matches and semantically similar content

class RetrieverBuilder:
    def __init__(self):
        """Initialize the retriever builder with embeddings."""
        embed_params = {
            EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: 3,
            EmbedTextParamsMetaNames.RETURN_OPTIONS: {"input_text": True},
        }

        # watsonx_embedding = WatsonxEmbeddings(
        #     model_id="ibm/slate-125m-english-rtrvr-v2",
        #     url="https://us-south.ml.cloud.ibm.com",
        #     project_id="skills-network",
        #     params=embed_params
        # )
        # self.embeddings = watsonx_embedding

        self.embeddings = OllamaEmbeddings(
            model = "nomic-embed-text"
        )


        
    def build_hybrid_retriever(self, docs):
        """Build a hybrid retriever using BM25 and vector-based retrieval."""
        try:
            # Create Chroma vector store
            # Stores documents embeddings using ChromaDB, and allows fast vector-based similarity search
            vector_store = Chroma.from_documents(
                documents=docs,
                embedding=self.embeddings,
                persist_directory=settings.CHROMA_DB_PATH
            )
            logger.info("Vector store created successfully.")
            
            # Create BM25 retriever which uses TF-IDF scoring and ranks documents based on keyword relevance.
            bm25 = BM25Retriever.from_documents(docs)
            logger.info("BM25 retriever created successfully.")
            
            # Create vector-based retriever, which retrieve documents based on vector similarity and returns top-k most relevant results
            vector_retriever = vector_store.as_retriever(search_kwargs={"k": settings.VECTOR_SEARCH_K})
            logger.info("Vector retriever created successfully.")
            
            # Combine retrievers into a hybrid retriever
            hybrid_retriever = EnsembleRetriever(
                retrievers=[bm25, vector_retriever],
                weights=settings.HYBRID_RETRIEVER_WEIGHTS #Adjust importance of lexical vs. vector search
            )
            logger.info("Hybrid retriever created successfully.")
            return hybrid_retriever
        except Exception as e:
            logger.error(f"Failed to build hybrid retriever: {e}")
            raise