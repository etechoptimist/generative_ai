# HomeMatch: Personalized Real Estate Listings using RAG


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

![RAG Architecture](https://assets.digitalocean.com/articles/alligator/boo.svg)

