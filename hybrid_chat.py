import json
from typing import Any, Dict, List, Optional
import google.generativeai as genai  # type: ignore
from pinecone import Pinecone  # type: ignore
from neo4j import GraphDatabase  # type: ignore
import config
from log import logger
import os

EMBED_MODEL = "gemini-embedding-001"
CHAT_MODEL = "gemini-2.5-flash"
TOP_K = 5
SIMILARITY_THRESHOLD = 0.7


class HybridChat:
    
    def __init__(self, query: str):
        self.query = query
        
        # Initialize all connections
        self._initialize_clients()
        

    def _initialize_clients(self):
        """Initialize all API clients and database connections."""
        try:
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

            # Initialize Google Generative AI
            logger.info("[hybrid_chat.py] Initializing Google Generative AI client")
            self.model = genai.GenerativeModel(model_name=CHAT_MODEL)
            
            # Initialize Pinecone
            logger.info("[hybrid_chat.py] Initializing Pinecone client")
            self.pc = Pinecone(api_key=config.PINECONE_API_KEY)
            self.index = self.pc.Index(config.PINECONE_INDEX_NAME)
            
            # Initialize Neo4j
            logger.info("[hybrid_chat.py] Initializing Neo4j driver")
            self.driver = GraphDatabase.driver(
                config.NEO4J_URI,
                auth=(config.NEO4J_USERNAME, config.NEO4J_PASSWORD)
            )
            
            # Test Neo4j connection
            self.driver.verify_connectivity()
            logger.info("[hybrid_chat.py] All clients initialized successfully")
            
        except Exception as e:
            logger.error(f"[hybrid_chat.py] Failed to initialize clients: {str(e)}")
            self.close()  # Clean up any partially initialized connections
            raise

    def _embed_query(self, text: str) -> List[float]:
        try:
            genai.configure(api_key=config.GOOGLE_API_KEY)
            logger.info(f"[hybrid_chat.py] Generating embedding for text: {text[:100]}...")
            resp = genai.embed_content(
                model="gemini-embedding-001",
                content=text,
                output_dimensionality=1536
            )


            
            embedding = resp['embedding']
            logger.info(f"[hybrid_chat.py] Successfully generated embedding of dimension {len(embedding)}")
            return embedding
        except Exception as e:
            logger.error(f"[hybrid_chat.py] Failed to generate embedding: {str(e)}")
            raise ValueError("Embedding generation failed") from e

    def query_pinecone(self, top_k: int = TOP_K) -> List[Dict[str, Any]]:
        try:
            logger.info(f"[hybrid_chat.py] Querying Pinecone with top_k={top_k}")
            prompt = """You are part of a hybrid retrieval system with a knowledge graph (Neo4j) 
and vector store (Pinecone) containing tourism data in JSON format.

Your job is to rewrite user queries into specific, information-rich queries 
that can retrieve relevant data from this dataset.

Context examples:

            {
            "id": "city_hanoi",
            "type": "City",
            "name": "Hanoi",
            "region": "Northern Vietnam",
            "description": "Hanoi is located in Northern Vietnam. It’s known for its culture, food, heritage experiences, combining local culture, food, and history. Travelers often visit for authentic Vietnamese experiences, from exploring markets and temples to trying street food and scenic excursions.",
            "best_time_to_visit": "February to May",
            "tags": ["culture", "food", "heritage"],
            "semantic_text": "Hanoi offers a mix of culture, food, heritage attractions and is a must-visit for those seeking immersive travel experiences in Vietnam.",
            "connections": [
                {"relation": "Connected_To", "target": "city_hue"},
                {"relation": "Connected_To", "target": "city_nha_trang"}
            ]
            }


User query: {user_query}

Rewrite the query into a version that includes specific details or entities 
(such as city names, tags, or related topics) from the dataset to help 
retrieve the most relevant information.

Output only the rewritten query.
            """
            transform_query = self.model.generate_content(prompt.replace("{user_query}", self.query))
            transformed_embeddings = self._embed_query(transform_query.text.strip())
            response = self.index.query(
                vector=transformed_embeddings,
                top_k=top_k,
                include_metadata=True
            )
            
            # Filter by similarity threshold
            relevant_matches = [
                match for match in response.matches
                if match.score >= 0.7
            ]

            logger.info(f"Found {len(relevant_matches)} relevant matches above threshold 0.7")

            if not relevant_matches:
                logger.warning("No matches found above similarity threshold")
            
            return relevant_matches
            
        except Exception as e:
            logger.error(f"Pinecone query failed: {str(e)}")
            raise

    def generate_cypher_query(self) -> str:
        try:
            logger.info("Generating Cypher query from natural language")

            with open("prompt.txt", "r", encoding="utf-8") as f:
                prompt_template = f.read()
            
            prompt = prompt_template.replace("[user_query]", self.query)
            
            response = self.model.generate_content(prompt)
            cypher_query = response.text.strip()
            
            # Remove markdown code block formatting if present
            if cypher_query.startswith("```"):
                lines = cypher_query.split("\n")
                cypher_query = "\n".join(lines[1:-1]) if len(lines) > 2 else cypher_query
            
            logger.info(f"Generated Cypher query: {cypher_query}")
            return cypher_query
            
        except FileNotFoundError:
            logger.error("prompt.txt file not found")
            raise
        except Exception as e:
            logger.error(f"Failed to generate Cypher query: {str(e)}")
            raise

    def query_neo4j(self, cypher_query: str) -> List[Dict[str, Any]]:
        try:
            logger.info("Executing Cypher query on Neo4j")
            
            with self.driver.session() as session:
                result = session.run(cypher_query)
                records = [record.data() for record in result]
            
            logger.info(f"Neo4j query returned {len(records)} records")
            return records
            
        except Exception as e:
            logger.error(f"Neo4j query failed: {str(e)}")
            logger.error(f"Failed query: {cypher_query}")
            return []

    def format_context(self, pinecone_results: List[Dict], neo4j_results: List[Dict]) -> str:
        logger.info("Formatting context from database results")
        
        context_parts = []
        
        # Add Pinecone results
        if pinecone_results:
            context_parts.append("=== Vector Search Results ===")
            for i, match in enumerate(pinecone_results, 1):
                metadata = match.metadata
                score = match.score
                context_parts.append(f"\n[Document {i}] (Similarity: {score:.3f})")
                context_parts.append(f"{metadata.get('text', 'No text available')}")
        
        # Add Neo4j results
        if neo4j_results:
            context_parts.append("\n\n=== Graph Database Results ===")
            for i, record in enumerate(neo4j_results, 1):
                context_parts.append(f"\n[Record {i}]")
                context_parts.append(json.dumps(record, indent=2))
        
        # Handle case where no results were found
        if not pinecone_results and not neo4j_results:
            context_parts.append("No relevant information found in the databases.")
        
        context = "\n".join(context_parts)
        logger.info(f"Context formatted, total length: {len(context)} characters")
        return context

    def generate_response(self, context: str) -> str:
        try:
            logger.info("Generating final response")
            
            if "No relevant information found" in context:
                prompt = f"""The user asked: "{self.query}"

However, no relevant information was found in the knowledge base. Please provide a helpful response 
explaining that you don't have specific information to answer their question, and suggest they 
rephrase or provide more context.

Answer:"""
            else:
                prompt = f"""You are VietGuide — a helpful, polite, and knowledgeable AI travel assistant for Vietnam.

Your role is to answer user questions about travel in Vietnam using only the information retrieved from the Pinecone vector database and Neo4j graph database.

Core Behavior Rules:
1. Always use **both the retrieved context and the user query** to generate your response. Do not answer without referring to the retrieved information.
2. Never guess, assume, or invent any information that is not present in the context.
3. If the context does not contain the requested information, reply politely:
   "I’m sorry, I don’t have that information right now. Once the database is updated, I’ll be able to provide you with a detailed answer."
   Then add: "Would you like to know about nearby destinations or other travel tips instead?"
4. Respond in a friendly, natural, and respectful tone, as if you are a polite local travel guide.
5. Keep responses short and clear, unless the user asks for a detailed explanation.
6. When describing travel destinations, include only verified details from the context, such as:
   - Location or region
   - Key attractions
   - Local culture
   - Food specialties
   - Best time to visit
   - Travel tips
7. Do not include internal reasoning, system messages, or code in your replies.
8. If a question is unrelated to Vietnam or travel, respond:
   "I’m designed to help with travel information in Vietnam. Could you please ask something related to that?"
9. Never use asterisks (*) or any markdown-style formatting in your response.
10. If you need to list points or steps, use plain text with line breaks or spaces instead of special characters or bullets.

Example of how to list points correctly:
Ha Long Bay travel tips:
- Take a boat tour to explore limestone islands
- Visit Sung Sot Cave
- Try local seafood dishes

Do not use:
* Take a boat tour
* Visit Sung Sot Cave

Your goal is to make every response polite, context-aware, accurate, and easy to read in plain text.
Always combine the context with the user query when forming your answer.
### Retrieved
Context:
{context}

User Question: {self.query}

Answer:"""
            
            response = self.model.generate_content(prompt)
            answer = response.text.strip()
            
            logger.info("Response generated successfully")
            return answer
            
        except Exception as e:
            logger.error(f"Failed to generate response: {str(e)}")
            raise

    def st(self, use_graph: bool = True) -> str:
        try:
            logger.info(f"Starting hybrid RAG pipeline for query: {self.query}")
            
            # Query Pinecone
            pinecone_results = self.query_pinecone()
            
            # Query Neo4j if enabled
            neo4j_results = []
            if use_graph:
                try:
                    cypher_query = self.generate_cypher_query()
                    neo4j_results = self.query_neo4j(cypher_query)
                except Exception as e:
                    logger.warning(f"Neo4j query skipped due to error: {str(e)}")
            
            # Format context and generate response
            context = self.format_context(pinecone_results, neo4j_results)
            response = self.generate_response(context)
            
            logger.info("Hybrid RAG pipeline completed successfully")
            return response
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}")
            raise

    def close(self):
        """Clean up database connections."""
        if self.driver:
            try:
                self.driver.close()
                logger.info("Neo4j driver closed")
            except Exception as e:
                logger.error(f"Error closing Neo4j driver: {str(e)}")

    def __enter__(self):
        """Support context manager protocol."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up when exiting context manager."""
        self.close()

    def __del__(self):
        """Clean up connections when object is destroyed."""
        self.close()
