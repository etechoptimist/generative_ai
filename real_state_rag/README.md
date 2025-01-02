# HomeMatch: Personalized Real Estate Listings using RAG

## Prerequisites
- Python 3.8+
- OpenAI API key
- ChromaDB
- Required Python packages (see requirements.txt)

## Installation

- Create and activate a virtual environment (recommended)
- Install dependencies from requirements.txt
pip install -r requirements.txt



## Project Overview
HomeMatch is an  real estate agent that leverages RAG (Retrieval Augmented Generation) to create personalized property recommendations. It transforms standard property listings into  narratives that match buyers' specific preferences and needs.

## Core Features
- Natural language preference collection
- Semantic search using vector embeddings
- Personalized listing descriptions
- Preference-based matching
- Fact-preserving content generation

## RAG Implementation
This project implements the RAG architecture in the following way:

### Retrieval Component
1. **Document Processing**
   - Loads real estate listings from CSV
   - Converts listings into embeddings
   - Stores in ChromaDB vector database

2. **Vector Search**
   - Uses Maximum Marginal Relevance (MMR)
   - Balances relevance with result diversity

### Augmentation Component
1. **Context Enhancement**
   - Enriches search results with buyer preferences
   - Maintains factual integrity
   - Adds personalized insights

### Generation Component
1. **Personalized Content**
   - Creates buyer-specific descriptions
   - Highlights relevant features
   - Preserves original listing facts

## Technical Architecture


<figure>
    <img src="https://github.com/etechoptimist/generative_ai/blob/master/real_state_rag/assets/rag.jpg"
         alt="Taken from DeepLearning course: LangChain Chat with Your Data">
    <figcaption>Taken from DeepLearning course: LangChain Chat with Your Data.</figcaption>
</figure>

