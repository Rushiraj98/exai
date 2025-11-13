"""
Knowledge Base - Vector Database Integration with Qdrant

Stores and retrieves historical energy patterns, solutions, and insights
using semantic similarity search.
"""

from typing import List, Dict, Optional
import numpy as np
from datetime import datetime
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    print("Warning: Qdrant client not available. Install with: pip install qdrant-client")


class EnergyKnowledgeBase:
    """
    Vector knowledge base for storing and retrieving energy patterns,
    solutions, and historical insights.
    """

    def __init__(self, host: str = "localhost", port: int = 6333, use_memory: bool = True):
        """
        Initialize knowledge base connection

        Args:
            host: Qdrant server host
            port: Qdrant server port
            use_memory: Use in-memory storage for demo (no Qdrant needed)
        """
        self.use_memory = use_memory or not QDRANT_AVAILABLE

        if not self.use_memory and QDRANT_AVAILABLE:
            try:
                self.client = QdrantClient(host=host, port=port)
                self._initialize_collections()
            except Exception as e:
                print(f"Warning: Could not connect to Qdrant at {host}:{port}")
                print(f"Falling back to in-memory storage. Error: {e}")
                self.use_memory = True

        if self.use_memory:
            self.memory_store = {
                "energy_patterns": [],
                "solutions": [],
                "building_insights": []
            }

    def _initialize_collections(self):
        """Create Qdrant collections if they don't exist"""
        collections = ["energy_patterns", "solutions", "building_insights"]

        for collection_name in collections:
            try:
                self.client.get_collection(collection_name)
            except:
                # Collection doesn't exist, create it
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=384,  # Typical embedding size (e.g., all-MiniLM-L6-v2)
                        distance=Distance.COSINE
                    )
                )

    def store_pattern(self, pattern: Dict, embedding: List[float],
                     collection: str = "energy_patterns") -> str:
        """
        Store an energy pattern with its embedding

        Args:
            pattern: Pattern metadata and description
            embedding: Vector embedding of the pattern
            collection: Collection name

        Returns:
            Pattern ID
        """
        pattern_id = f"{collection}_{datetime.now().timestamp()}"

        if self.use_memory:
            self.memory_store[collection].append({
                "id": pattern_id,
                "vector": embedding,
                "payload": pattern
            })
        else:
            point = PointStruct(
                id=pattern_id,
                vector=embedding,
                payload=pattern
            )
            self.client.upsert(collection_name=collection, points=[point])

        return pattern_id

    def search_similar_patterns(self, query_embedding: List[float],
                               collection: str = "energy_patterns",
                               limit: int = 5,
                               filters: Optional[Dict] = None) -> List[Dict]:
        """
        Search for similar patterns using vector similarity

        Args:
            query_embedding: Query vector embedding
            collection: Collection to search
            limit: Number of results to return
            filters: Optional metadata filters

        Returns:
            List of similar patterns with scores
        """
        if self.use_memory:
            return self._search_memory(query_embedding, collection, limit)
        else:
            search_filter = None
            if filters:
                # Build Qdrant filter from dict
                search_filter = self._build_filter(filters)

            results = self.client.search(
                collection_name=collection,
                query_vector=query_embedding,
                limit=limit,
                query_filter=search_filter
            )

            return [
                {
                    "id": result.id,
                    "score": result.score,
                    "payload": result.payload
                }
                for result in results
            ]

    def store_solution(self, problem: str, solution: str, building_id: str,
                      effectiveness: float, embedding: List[float]) -> str:
        """
        Store a successful solution for future reference

        Args:
            problem: Problem description
            solution: Solution that was applied
            building_id: Building where solution was applied
            effectiveness: How effective it was (0-1)
            embedding: Vector embedding

        Returns:
            Solution ID
        """
        solution_data = {
            "problem": problem,
            "solution": solution,
            "building_id": building_id,
            "effectiveness": effectiveness,
            "timestamp": datetime.now().isoformat(),
            "applicable_building_types": []  # Would be populated with analysis
        }

        return self.store_pattern(solution_data, embedding, collection="solutions")

    def get_relevant_solutions(self, problem_embedding: List[float],
                              building_type: Optional[str] = None) -> List[Dict]:
        """
        Find relevant solutions from past experiences

        Args:
            problem_embedding: Embedding of current problem
            building_type: Optional filter by building type

        Returns:
            List of relevant historical solutions
        """
        filters = None
        if building_type:
            filters = {"building_type": building_type}

        solutions = self.search_similar_patterns(
            problem_embedding,
            collection="solutions",
            limit=5,
            filters=filters
        )

        # Sort by effectiveness
        solutions.sort(key=lambda x: x.get('payload', {}).get('effectiveness', 0), reverse=True)

        return solutions

    def store_building_insight(self, building_id: str, insight: str,
                              category: str, embedding: List[float]) -> str:
        """
        Store an insight about a building

        Args:
            building_id: Building identifier
            insight: Insight text
            category: Insight category (e.g., 'hvac', 'envelope', 'occupancy')
            embedding: Vector embedding

        Returns:
            Insight ID
        """
        insight_data = {
            "building_id": building_id,
            "insight": insight,
            "category": category,
            "timestamp": datetime.now().isoformat()
        }

        return self.store_pattern(insight_data, embedding, collection="building_insights")

    def _search_memory(self, query_embedding: List[float], collection: str, limit: int) -> List[Dict]:
        """Search in-memory store using cosine similarity"""
        items = self.memory_store.get(collection, [])

        if not items:
            return []

        # Calculate cosine similarities
        query_vec = np.array(query_embedding)
        scores = []

        for item in items:
            item_vec = np.array(item['vector'])
            # Cosine similarity
            similarity = np.dot(query_vec, item_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(item_vec))
            scores.append({
                "id": item['id'],
                "score": float(similarity),
                "payload": item['payload']
            })

        # Sort by score and return top results
        scores.sort(key=lambda x: x['score'], reverse=True)
        return scores[:limit]

    def _build_filter(self, filters: Dict) -> Filter:
        """Build Qdrant filter from dictionary"""
        conditions = []
        for key, value in filters.items():
            conditions.append(
                FieldCondition(
                    key=key,
                    match=MatchValue(value=value)
                )
            )

        return Filter(must=conditions) if conditions else None

    def get_collection_stats(self, collection: str = "energy_patterns") -> Dict:
        """Get statistics about a collection"""
        if self.use_memory:
            return {
                "collection": collection,
                "total_items": len(self.memory_store.get(collection, [])),
                "storage_type": "in-memory"
            }
        else:
            try:
                info = self.client.get_collection(collection)
                return {
                    "collection": collection,
                    "total_items": info.points_count,
                    "vector_size": info.config.params.vectors.size,
                    "storage_type": "qdrant"
                }
            except Exception as e:
                return {"error": str(e)}


class SimpleEmbedder:
    """
    Simple text embedder using sentence transformers
    (In production, use proper embedding models)
    """

    def __init__(self):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.available = True
        except ImportError:
            print("Warning: sentence-transformers not available. Using random embeddings.")
            self.available = False

    def embed(self, text: str) -> List[float]:
        """
        Generate embedding for text

        Args:
            text: Input text

        Returns:
            Embedding vector
        """
        if self.available:
            embedding = self.model.encode(text)
            return embedding.tolist()
        else:
            # Return random embedding for demo
            return np.random.randn(384).tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts

        Args:
            texts: List of input texts

        Returns:
            List of embedding vectors
        """
        if self.available:
            embeddings = self.model.encode(texts)
            return embeddings.tolist()
        else:
            # Return random embeddings for demo
            return [np.random.randn(384).tolist() for _ in texts]


def populate_demo_knowledge_base(kb: EnergyKnowledgeBase):
    """
    Populate knowledge base with demo patterns and solutions

    Args:
        kb: Knowledge base instance
    """
    embedder = SimpleEmbedder()

    # Demo patterns
    demo_patterns = [
        {
            "description": "High afternoon cooling load in glass-facade building",
            "building_type": "office",
            "severity": "high",
            "common_causes": ["solar gain", "poor insulation", "undersized HVAC"]
        },
        {
            "description": "Excessive nighttime consumption in residential tower",
            "building_type": "residential",
            "severity": "medium",
            "common_causes": ["HVAC scheduling issue", "phantom loads", "inefficient lighting"]
        },
        {
            "description": "Morning pre-cooling inefficiency",
            "building_type": "office",
            "severity": "medium",
            "common_causes": ["wrong pre-cool timing", "insufficient capacity", "poor control logic"]
        }
    ]

    for pattern in demo_patterns:
        embedding = embedder.embed(pattern["description"])
        kb.store_pattern(pattern, embedding)

    # Demo solutions
    demo_solutions = [
        {
            "problem": "High solar gain on west facade",
            "solution": "Automated blind control + pre-cooling optimization",
            "building_id": "Marina_Tower_2",
            "effectiveness": 0.89
        },
        {
            "problem": "HVAC running during low occupancy",
            "solution": "Occupancy-based scheduling with 30-min ramp-up",
            "building_id": "JLT_Tower_5",
            "effectiveness": 0.76
        }
    ]

    for solution in demo_solutions:
        embedding = embedder.embed(solution["problem"])
        kb.store_solution(
            problem=solution["problem"],
            solution=solution["solution"],
            building_id=solution["building_id"],
            effectiveness=solution["effectiveness"],
            embedding=embedding
        )

    print(f"✅ Populated knowledge base with {len(demo_patterns)} patterns and {len(demo_solutions)} solutions")
