from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Request # type: ignore
from fastapi.middleware.cors import CORSMiddleware # type: ignore
from fastapi.responses import FileResponse, JSONResponse    # type: ignore
from fastapi.staticfiles import StaticFiles # type: ignore
from pydantic import BaseModel # type: ignore
from typing import Optional, List, Dict, Any, Tuple
import asyncio
import google.generativeai as genai  # type: ignore
from langchain_community.vectorstores import FAISS # type: ignore
from langchain_core.documents import Document # type: ignore
from langchain_huggingface import HuggingFaceEmbeddings # type: ignore
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import time
from datetime import datetime
from contextlib import asynccontextmanager
import os
import numpy as np
import pickle
from pathlib import Path

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Import Redis (handle if not installed)
try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    print("Warning: redis not installed. Caching will be disabled.")
    print("Install with: pip install redis")

# Import your existing modules
from hybrid_chat import HybridChat
from log import logger
import config

# Global variables
executor = ThreadPoolExecutor(max_workers=20)  # Increased for parallel operations
redis_client: Optional[Any] = None
faiss_cache: Optional[FAISS] = None
faiss_metadata: Dict[str, Any] = {}
active_connections: List[WebSocket] = []
faiss_lock = asyncio.Lock()

# Configuration
MAX_REQUESTS_PER_MINUTE = 20
REDIS_CACHE_TTL = 3600  # 1 hour for exact matches
FAISS_SIMILARITY_THRESHOLD = 0.95  # 95% similarity for cache hit
FAISS_CACHE_DIR = "faiss_cache"
FAISS_MAX_ENTRIES = 10000  # Maximum cached queries in FAISS

# Request/Response models
class QueryRequest(BaseModel):
    query: str
    user_id: Optional[str] = "anonymous"
    use_graph: bool = True

class QueryResponse(BaseModel):
    response: str
    cached: bool
    cache_type: Optional[str] = None  # "exact" or "similar"
    similarity_score: Optional[float] = None
    processing_time: float
    sources: Dict[str, int]

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    redis: bool
    faiss: bool
    version: str

# Rate limiting storage (in-memory for simplicity)
user_request_counts: Dict[str, List[float]] = {}

def init_faiss_cache():
    """Initialize FAISS cache from disk or create new one."""
    global faiss_cache, faiss_metadata
    
    Path(FAISS_CACHE_DIR).mkdir(exist_ok=True)
    faiss_path = Path(FAISS_CACHE_DIR) / "index"
    metadata_path = Path(FAISS_CACHE_DIR) / "metadata.pkl"
    
    try:
        if faiss_path.exists() and metadata_path.exists():
            logger.info("   Loading existing FAISS cache...")
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            faiss_cache = FAISS.load_local(
                str(faiss_path), 
                embeddings,
                allow_dangerous_deserialization=True
            )
            with open(metadata_path, 'rb') as f:
                faiss_metadata = pickle.load(f)
            logger.info(f"   FAISS cache loaded: {len(faiss_metadata)} entries")
        else:
            logger.info("   Creating new FAISS cache...")
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            # Initialize with dummy document
            dummy_doc = Document(page_content="init", metadata={"query_hash": "init"})
            faiss_cache = FAISS.from_documents([dummy_doc], embeddings)
            faiss_metadata = {}
            logger.info("   New FAISS cache created")
    except Exception as e:
        logger.error(f"   Failed to initialize FAISS cache: {e}")
        faiss_cache = None

def save_faiss_cache():
    """Save FAISS cache to disk."""
    if faiss_cache is None:
        return
    
    try:
        faiss_path = Path(FAISS_CACHE_DIR) / "index"
        metadata_path = Path(FAISS_CACHE_DIR) / "metadata.pkl"
        
        faiss_cache.save_local(str(faiss_path))
        with open(metadata_path, 'wb') as f:
            pickle.dump(faiss_metadata, f)
        logger.info(f"   FAISS cache saved: {len(faiss_metadata)} entries")
    except Exception as e:
        logger.error(f"   Failed to save FAISS cache: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    global redis_client, faiss_cache
    
    # Startup
    logger.info("    Starting FastAPI backend...")
    logger.info(f"   Max workers: {executor._max_workers}")
    logger.info(f"   Rate limit: {MAX_REQUESTS_PER_MINUTE} requests/minute")
    
    # Try to connect to Redis
    if REDIS_AVAILABLE:
        try:
            redis_client = await redis.from_url(
                "redis://localhost:6379",
                encoding="utf-8",
                decode_responses=True
            )
            await redis_client.ping()
            logger.info(" Redis connected successfully")
        except Exception as e:
            logger.warning(f"  Redis connection failed: {e}")
            logger.warning("   Caching will be disabled")
            redis_client = None
    else:
        logger.warning("  Redis not available. Install with: pip install redis")
    
    # Initialize FAISS cache
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(executor, init_faiss_cache)
    
    logger.info("    Backend started successfully")
    logger.info("    Access frontend at: http://localhost:8000")

    yield
    
    # Shutdown
    logger.info("    Shutting down FastAPI backend...")
    
    # Save FAISS cache
    if faiss_cache:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(executor, save_faiss_cache)
    
    if redis_client:
        await redis_client.close()
        logger.info("   Redis connection closed")
    executor.shutdown(wait=True)
    logger.info("   Thread pool closed")
    logger.info("   Shutdown complete")

# Initialize FastAPI app
app = FastAPI(
    title="Vietnam Travel Assistant API",
    description="Hybrid RAG system with Neo4j and Pinecone + Dual Cache (Redis + FAISS)",
    version="2.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Helper functions
def get_query_hash(query: str) -> str:
    """Generate MD5 hash from query for exact matching."""
    return hashlib.md5(query.encode('utf-8')).hexdigest()

def get_embedding(query: str) -> np.ndarray:
    """Generate embedding for query using Gemini."""
    genai.configure(api_key=config.GOOGLE_API_KEY)
    resp = genai.embed_content(
        model="gemini-embedding-001",
        content=query,
        output_dimensionality=1536
    )
    return np.array(resp['embedding'])

async def get_redis_cache(query_hash: str) -> Optional[Dict]:
    """Get exact match from Redis cache."""
    if not redis_client:
        return None
    try:
        cached = await redis_client.get(f"exact:{query_hash}")
        if cached:
            logger.info(f"   Redis HIT (exact): {query_hash[:16]}...")
            return json.loads(cached)
        return None
    except Exception as e:
        logger.error(f"Redis read error: {e}")
        return None

async def set_redis_cache(query_hash: str, data: Dict, ttl: int = REDIS_CACHE_TTL):
    """Set exact match in Redis cache."""
    if not redis_client:
        return
    try:
        await redis_client.setex(
            f"exact:{query_hash}",
            ttl,
            json.dumps(data)
        )
        logger.info(f"   Redis SET: {query_hash[:16]}... (TTL: {ttl}s)")
    except Exception as e:
        logger.error(f"Redis write error: {e}")

async def search_faiss_cache(query: str, embedding: np.ndarray) -> Optional[Tuple[Dict, float]]:
    """Search for similar query in FAISS cache."""
    if faiss_cache is None or len(faiss_metadata) == 0:
        return None
    
    async with faiss_lock:
        try:
            # Run FAISS search in thread pool
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                executor,
                lambda: faiss_cache.similarity_search_with_score_by_vector(
                    embedding.tolist(),
                    k=1
                )
            )
            
            if results and len(results) > 0:
                doc, distance = results[0]
                # Convert L2 distance to similarity score (0-1)
                similarity = 1 / (1 + distance)
                
                if similarity >= 0.7:
                    query_hash = doc.metadata.get("query_hash")
                    if query_hash in faiss_metadata:
                        logger.info(f"   FAISS HIT (similar): score={similarity:.3f}")
                        return faiss_metadata[query_hash], similarity
            
            return None
        except Exception as e:
            logger.error(f"FAISS search error: {e}")
            return None

async def add_to_faiss_cache(query: str, query_hash: str, embedding: np.ndarray, data: Dict):
    """Add query and response to FAISS cache."""
    if faiss_cache is None:
        return
    
    async with faiss_lock:
        try:
            # Check if we need to limit cache size
            if len(faiss_metadata) >= FAISS_MAX_ENTRIES:
                logger.warning(f"   FAISS cache full ({FAISS_MAX_ENTRIES}), skipping add")
                return
            
            # Add to FAISS in thread pool
            loop = asyncio.get_event_loop()
            doc = Document(
                page_content=query,
                metadata={"query_hash": query_hash}
            )
            
            await loop.run_in_executor(
                executor,
                lambda: faiss_cache.add_documents([doc])
            )
            
            # Store metadata
            faiss_metadata[query_hash] = {
                "query": query,
                "data": data,
                "timestamp": time.time()
            }
            
            logger.info(f"   FAISS ADD: {query_hash[:16]}... (total: {len(faiss_metadata)})")
            
            # Periodic save (every 10 entries)
            if len(faiss_metadata) % 10 == 0:
                await loop.run_in_executor(executor, save_faiss_cache)
                
        except Exception as e:
            logger.error(f"FAISS add error: {e}")

def check_rate_limit(user_id: str) -> bool:
    """Check if user has exceeded rate limit."""
    now = time.time()
    
    if user_id in user_request_counts:
        user_request_counts[user_id] = [
            t for t in user_request_counts[user_id] 
            if now - t < 60
        ]
    else:
        user_request_counts[user_id] = []
    
    if len(user_request_counts[user_id]) >= MAX_REQUESTS_PER_MINUTE:
        logger.warning(f"     Rate limit exceeded for user: {user_id}")
        return False
    
    user_request_counts[user_id].append(now)
    return True

async def run_parallel_queries(chat: HybridChat, use_graph: bool = True) -> tuple:
    """Run Neo4j and Pinecone queries in parallel using ThreadPoolExecutor."""
    
    def run_neo4j_query():
        if not use_graph:
            return []
        try:
            cypher_query = chat.generate_cypher_query()
            return chat.query_neo4j(cypher_query)
        except Exception as e:
            logger.error(f"Neo4j query failed: {str(e)}")
            return []
    
    def run_pinecone_query():
        try:
            return chat.query_pinecone()
        except Exception as e:
            logger.error(f"Pinecone query failed: {str(e)}")
            return []
    
    logger.info("   Running parallel queries...")
    loop = asyncio.get_event_loop()
    neo4j_future = loop.run_in_executor(executor, run_neo4j_query)
    pinecone_future = loop.run_in_executor(executor, run_pinecone_query)
    
    neo4j_results, pinecone_results = await asyncio.gather(
        neo4j_future,
        pinecone_future,
        return_exceptions=True
    )
    
    if isinstance(neo4j_results, Exception):
        logger.error(f"Neo4j exception: {str(neo4j_results)}")
        neo4j_results = []
    
    if isinstance(pinecone_results, Exception):
        logger.error(f"Pinecone exception: {str(pinecone_results)}")
        pinecone_results = []
    
    logger.info(f"   Results: Neo4j={len(neo4j_results)}, Pinecone={len(pinecone_results)}")
    return neo4j_results, pinecone_results

# Routes
@app.get("/")
async def serve_frontend():
    """Serve the frontend HTML file."""
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    else:
        return {
            "error": "index.html not found",
            "message": "Please create index.html in the same directory as main.py"
        }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        redis=redis_client is not None,
        faiss=faiss_cache is not None,
        version="2.0.0"
    )

@app.post("/api/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    Process user query with dual-layer caching (Redis + FAISS).
    
    Cache Strategy:
    1. Check Redis for exact query match (MD5 hash)
    2. If not found, check FAISS for similar query (embedding similarity)
    3. If not found, process query and cache in both Redis and FAISS
    """
    start_time = time.time()
    
    logger.info("=" * 80)
    logger.info(f"   NEW QUERY from {request.user_id}")
    logger.info(f"   Query: {request.query[:100]}...")
    
    # Rate limiting
    if not check_rate_limit(request.user_id):
        logger.warning(f"    Rate limit exceeded for {request.user_id}")
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Maximum {MAX_REQUESTS_PER_MINUTE} requests per minute."
        )
    
    # Generate query hash for exact matching
    query_hash = get_query_hash(request.query)
    
    # Step 1: Check Redis for exact match
    redis_result = await get_redis_cache(query_hash)
    if redis_result:
        processing_time = time.time() - start_time
        logger.info(f"   Returned from Redis cache in {processing_time:.2f}s")
        logger.info("=" * 80)
        return QueryResponse(
            response=redis_result["response"],
            cached=True,
            cache_type="exact",
            processing_time=processing_time,
            sources=redis_result["sources"]
        )
    
    # Step 2: Generate embedding and check FAISS for similar queries
    logger.info("   Redis MISS - Generating embedding...")
    loop = asyncio.get_event_loop()
    embedding = await loop.run_in_executor(executor, get_embedding, request.query)
    
    faiss_result = await search_faiss_cache(request.query, embedding)
    if faiss_result:
        cached_data, similarity = faiss_result
        processing_time = time.time() - start_time
        logger.info(f"   Returned from FAISS cache in {processing_time:.2f}s")
        logger.info("=" * 80)
        
        # Also cache in Redis for faster future access
        await set_redis_cache(query_hash, cached_data["data"])
        
        return QueryResponse(
            response=cached_data["data"]["response"],
            cached=True,
            cache_type="similar",
            similarity_score=similarity,
            processing_time=processing_time,
            sources=cached_data["data"]["sources"]
        )
    
    # Step 3: Process query (cache miss)
    try:
        logger.info("    Both caches MISS - Processing query...")
        
        # Initialize HybridChat
        chat = HybridChat(query=request.query)
        
        # Run parallel queries
        neo4j_results, pinecone_results = await run_parallel_queries(
            chat, 
            use_graph=request.use_graph
        )
        
        # Format context and generate response in parallel with cleanup
        logger.info("   Generating response...")
        
        def generate_and_cleanup():
            context = chat.format_context(pinecone_results, neo4j_results)
            response = chat.generate_response(context)
            chat.close()
            return response
        
        response = await loop.run_in_executor(executor, generate_and_cleanup)
        
        # Prepare response data
        sources = {
            "neo4j": len(neo4j_results),
            "pinecone": len(pinecone_results)
        }
        
        response_data = {
            "response": response,
            "sources": sources
        }
        
        # Cache in both Redis and FAISS in parallel
        await asyncio.gather(
            set_redis_cache(query_hash, response_data),
            add_to_faiss_cache(request.query, query_hash, embedding, response_data)
        )
        
        processing_time = time.time() - start_time
        logger.info(f"   Query processed successfully in {processing_time:.2f}s")
        logger.info("=" * 80)
        
        return QueryResponse(
            response=response,
            cached=False,
            processing_time=processing_time,
            sources=sources
        )
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"   Query processing failed after {processing_time:.2f}s")
        logger.error(f"   Error: {str(e)}")
        logger.error("=" * 80)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process query: {str(e)}"
        )

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time chat."""
    await websocket.accept()
    active_connections.append(websocket)
    logger.info(f"🔌 WebSocket connected (total: {len(active_connections)})")
    
    try:
        while True:
            data = await websocket.receive_text()
            query_data = json.loads(data)
            
            logger.info(f"   WebSocket query: {query_data.get('query', '')[:50]}...")
            await websocket.send_json({"status": "processing"})
            
            try:
                chat = HybridChat(query=query_data["query"])
                neo4j_results, pinecone_results = await run_parallel_queries(chat)
                context = chat.format_context(pinecone_results, neo4j_results)
                response = chat.generate_response(context)
                chat.close()
                
                await websocket.send_json({
                    "status": "complete",
                    "response": response,
                    "sources": {
                        "neo4j": len(neo4j_results),
                        "pinecone": len(pinecone_results)
                    }
                })
                
                logger.info("    WebSocket response sent")
                
            except Exception as e:
                logger.error(f"   WebSocket error: {str(e)}")
                await websocket.send_json({
                    "status": "error",
                    "error": str(e)
                })
                
    except WebSocketDisconnect:
        active_connections.remove(websocket)
        logger.info(f"🔌 WebSocket disconnected (remaining: {len(active_connections)})")

@app.get("/api/stats")
async def get_stats():
    """Get system statistics."""
    cache_info = {}
    if redis_client:
        try:
            info = await redis_client.info()
            cache_info = {
                "used_memory_human": info.get("used_memory_human", "unknown"),
                "connected_clients": info.get("connected_clients", 0),
                "total_commands_processed": info.get("total_commands_processed", 0)
            }
        except Exception as e:
            logger.error(f"Failed to get Redis stats: {e}")
    
    faiss_stats = {
        "enabled": faiss_cache is not None,
        "total_entries": len(faiss_metadata),
        "max_entries": FAISS_MAX_ENTRIES,
        "similarity_threshold": FAISS_SIMILARITY_THRESHOLD
    }
    
    return {
        "active_websockets": len(active_connections),
        "total_users_tracked": len(user_request_counts),
        "redis_cache": {
            "enabled": redis_client is not None,
            "info": cache_info
        },
        "faiss_cache": faiss_stats,
        "executor_threads": executor._max_workers,
        "rate_limit": MAX_REQUESTS_PER_MINUTE
    }

@app.delete("/api/cache/clear")
async def clear_cache():
    """Clear all cached responses (both Redis and FAISS)."""
    cleared = {"redis": 0, "faiss": 0}
    
    # Clear Redis
    if redis_client:
        try:
            keys = []
            async for key in redis_client.scan_iter(match="exact:*"):
                keys.append(key)
            
            if keys:
                await redis_client.delete(*keys)
                cleared["redis"] = len(keys)
                logger.info(f"    Cleared {len(keys)} Redis entries")
        except Exception as e:
            logger.error(f"Failed to clear Redis cache: {e}")
    
    # Clear FAISS
    global faiss_cache, faiss_metadata
    if faiss_cache:
        try:
            async with faiss_lock:
                cleared["faiss"] = len(faiss_metadata)
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(executor, init_faiss_cache)
                logger.info(f"    Cleared {cleared['faiss']} FAISS entries")
        except Exception as e:
            logger.error(f"Failed to clear FAISS cache: {e}")
    
    return {
        "message": "Cache cleared",
        "redis_entries_cleared": cleared["redis"],
        "faiss_entries_cleared": cleared["faiss"]
    }

@app.get("/api/cache/stats")
async def cache_stats():
    """Get detailed cache statistics."""
    redis_stats = {"enabled": False}
    if redis_client:
        try:
            keys = []
            async for key in redis_client.scan_iter(match="exact:*"):
                keys.append(key)
            redis_stats = {
                "enabled": True,
                "total_cached_queries": len(keys),
                "ttl": REDIS_CACHE_TTL
            }
        except Exception as e:
            logger.error(f"Failed to get Redis stats: {e}")
    
    faiss_stats = {
        "enabled": faiss_cache is not None,
        "total_entries": len(faiss_metadata),
        "max_entries": FAISS_MAX_ENTRIES,
        "similarity_threshold": FAISS_SIMILARITY_THRESHOLD,
        "usage_percent": (len(faiss_metadata) / FAISS_MAX_ENTRIES * 100) if FAISS_MAX_ENTRIES > 0 else 0
    }
    
    return {
        "redis": redis_stats,
        "faiss": faiss_stats
    }

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Handle 404 errors."""
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not found",
            "message": "The requested endpoint does not exist",
            "available_endpoints": [
                "GET /",
                "GET /health",
                "POST /api/query",
                "GET /api/stats",
                "GET /api/cache/stats",
                "DELETE /api/cache/clear",
                "WS /ws"
            ]
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "Something went wrong. Please check the logs."
        }
    )

if __name__ == "__main__":
    import uvicorn
    
    if not os.path.exists("index.html"):
        print("\n" + "⚠️  " + "=" * 76)
        print("⚠️  WARNING: index.html not found!")
        print("⚠️  " + "=" * 76)
        print("⚠️  Please create index.html in the same directory as main.py")
        print("⚠️  The frontend will not work without it.")
        print("⚠️  " + "=" * 76 + "\n")
    
    print("\n" + "=" * 80)
    print("   VIETNAM TRAVEL ASSISTANT - DUAL CACHE SYSTEM")
    print("=" * 80)
    print(f"   Frontend URL: http://localhost:8000")
    print(f"   API Docs: http://localhost:8000/docs")
    print(f"   Health Check: http://localhost:8000/health")
    print(f"   Stats: http://localhost:8000/api/stats")
    print("=" * 80)
    print(f"     Configuration:")
    print(f"   - Max Workers: {executor._max_workers}")
    print(f"   - Rate Limit: {MAX_REQUESTS_PER_MINUTE} requests/minute")
    print(f"   - Redis TTL: {REDIS_CACHE_TTL} seconds")
    print(f"   - FAISS Threshold: {FAISS_SIMILARITY_THRESHOLD}")
    print(f"   - FAISS Max Entries: {FAISS_MAX_ENTRIES}")
    print("=" * 80)
    print("💡 Dual Cache Strategy:")
    print("   1. Redis: Exact query matching (MD5 hash)")
    print("   2. FAISS: Semantic similarity search (95%+ match)")
    print("=" * 80 + "\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True
    )