# DocChat: Build a Multi-Agent RAG System

## Project Overview

DocChat is a multi-agent application, that explores the power of multi-agent retrieval-augmented generation (RAG) using LangGraph, Docling, Ollama-powered LLMs, and ChromaDB. This project integrates multiple components—document processing, hybrid retrieval, structured AI workflows, and verification agents—into a seamless, interactive AI-driven application.

With the Gradio-powered UI, users can intuitively upload documents, ask complex questions, and receive fact-checked, AI-generated responses in real time. This marks the culmination of combining retrieval techniques, structured AI interactions, and LLM-based response generation into a production-ready system.

![image.png](attachment:image.png)

## 1 - User query processing & relevance analysis
* The system starts when a user submits a question about their uploaded document(s)
* Before retrieving any data, DocChat first analyzes query relevance to determine if the question is within the scope of the uploaded content

## 2 - Routing & query categorization
* The query is routed through an intelligent agent that decides whether the system can answer it using the document(s):
 * In scope: Proceed with document retrieval and response generation.
 * Not in scope: Inform the user that the question cannot be answered based on the provided documents, preventing hallucinations.

## 3 - Multi-agent research & document retrieval
* If the query is relevant, DocChat retrieves relevant document sections from a hybrid search system:
 - Docling converts the document into a structured Markdown format for better chunking
* LangChain splits the document into logical chunks based on headers and stores them in ChromaDB (a vector store)
* The retrieval module searches for the most contextually relevant document chunks using BM25 and vector search

## 4 - Answer generation & verification loop
Conduct research:

* The research agent generates an initial answer based on retrieved content
* A sub-process starts where queries are dynamically generated for more precise retrieval

Verification process:

* The verification agent cross-checks the generated response against the retrieved content
* If the response is fully supported, the system finalizes and returns the answer
* If verification fails (e.g., hallucinations, unsupported claims), the system re-runs the research step until a verifiable response is found
## 5 - Response finalization
* After verification is complete, DocChat returns the final response to the user
* The workflow ensures that each answer is sourced directly from the provided document(s), preventing fabrication or unreliable outputs