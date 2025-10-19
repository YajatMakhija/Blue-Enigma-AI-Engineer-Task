import json
from typing import Dict, List, Optional, Any
from neo4j import GraphDatabase  # type: ignore
from neo4j.exceptions import ServiceUnavailable, TransientError, ClientError  # type: ignore
from tqdm import tqdm  # type: ignore
import config
from log import logger
from pyvis.network import Network # type: ignore
import networkx as nx # type: ignore
import config
NEO_BATCH = 500

DATA_FILE = "vietnam_travel_dataset.json"


class Neo4jClient:
    MAX_RETRIES = 3
    BATCH_SIZE = 100  # Process nodes in batches for better performance
    
    def __init__(self, data: str = DATA_FILE):
        self.data = data
        self.driver = None
        self._connect()
    
    def _connect(self) -> None:
        """Establish connection to Neo4j database."""
        try:
            logger.info(f"[load_to_neo4j.py] Connecting to Neo4j at {config.NEO4J_URI}")
            self.driver = GraphDatabase.driver(
                config.NEO4J_URI,
                auth=(config.NEO4J_USERNAME, config.NEO4J_PASSWORD)
            )
            # Test connection
            self.driver.verify_connectivity()
            logger.info("[load_to_neo4j.py] Successfully connected to Neo4j database")
        except ServiceUnavailable as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during connection: {e}")
            raise 
    
    def close(self) -> None:
        """Close the database connection."""
        if self.driver:
            try:
                self.driver.close()
                logger.info("[load_to_neo4j.py] Neo4j connection closed")
            except Exception as e:
                logger.error(f"[load_to_neo4j.py] Error closing connection: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    def _load_data(self) -> List[Dict[str, Any]]:
        try:
            logger.info(f"Loading data from {self.data}")
            with open(self.data, "r", encoding="utf-8") as f:
                nodes = json.load(f)
            logger.info(f"Successfully loaded {len(nodes)} nodes")
            return nodes
        except FileNotFoundError:
            logger.error(f"Data file not found: {self.data}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in data file: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def _validate_node(self, node: Dict[str, Any]) -> bool:
        """
        Validate node has required fields.
        
        Args:
            node: Node dictionary to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not isinstance(node, dict):
            logger.warning("Node is not a dictionary")
            return False
        
        if "id" not in node:
            logger.warning("Node missing required 'id' field")
            return False
        
        if not node.get("type"):
            logger.debug(f"Node {node['id']} missing 'type' field, will use 'Unknown'")
        
        return True
    
    def upload_data(self) -> None:
        """Upload all data to Neo4j database."""
        logger.info("Starting data upload process")
        
        nodes = self._load_data()
        
        if not nodes:
            logger.warning("No nodes to upload")
            return
        
        # Validate nodes
        valid_nodes = []
        for node in nodes:
            if self._validate_node(node):
                valid_nodes.append(node)
            else:
                logger.warning(f"Skipping invalid node: {node.get('id', 'unknown')}")
        
        logger.info(f"Validated {len(valid_nodes)}/{len(nodes)} nodes")
        
        try:
            with self.driver.session() as session: 
                # Create constraints first
                logger.info("Creating database constraints")
                session.execute_write(self._create_constraints)
                
                # Create nodes with progress bar
                logger.info("Creating nodes in Neo4j")
                failed_nodes = []
                for node in tqdm(valid_nodes, desc="Creating nodes"):
                    success = self._upsert_node_with_retry(session, node)
                    if not success:
                        failed_nodes.append(node["id"])
                
                if failed_nodes:
                    logger.warning(f"Failed to create {len(failed_nodes)} nodes: {failed_nodes[:10]}")
                
                # Create relationships with progress bar
                logger.info("Creating relationships in Neo4j")
                failed_rels = []
                for node in tqdm(valid_nodes, desc="Creating relationships"):
                    conns = node.get("connections", [])
                    if not conns:
                        logger.debug(f"Node {node['id']} has no connections")
                        continue
                    
                    for rel in conns:
                        success = self._create_relationship_with_retry(session, node["id"], rel)
                        if not success:
                            failed_rels.append((node["id"], rel.get("target")))
                
                if failed_rels:
                    logger.warning(f"Failed to create {len(failed_rels)} relationships")
                
                logger.info("Data upload completed successfully")
                
        except ServiceUnavailable as e:
            logger.error(f"Database service unavailable: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during upload: {e}")
            raise
    
    def _upsert_node_with_retry(self, session, node: Dict[str, Any]) -> bool:
        """
        Upsert node with retry logic.
        
        Args:
            session: Neo4j session
            node: Node data dictionary
            
        Returns:
            True if successful, False otherwise
        """
        for attempt in range(self.MAX_RETRIES):
            try:
                session.execute_write(self._upsert_node, node)
                logger.debug(f"Successfully created/updated node: {node['id']}")
                return True
            except TransientError as e:
                logger.warning(f"Transient error on attempt {attempt + 1} for node {node['id']}: {e}")
                if attempt == self.MAX_RETRIES - 1:
                    logger.error(f"Failed to create node {node['id']} after {self.MAX_RETRIES} attempts")
                    return False
            except ClientError as e:
                logger.error(f"Client error creating node {node['id']}: {e}")
                return False
            except Exception as e:
                logger.error(f"Unexpected error creating node {node['id']}: {e}")
                return False
        
        return False
    
    def _upsert_node(self, tx, node: Dict[str, Any]) -> None:
        """
        Create or update a node in Neo4j.
        
        Args:
            tx: Neo4j transaction
            node: Node data dictionary
        """
        try:
            labels = [node.get("type", "Unknown"), "Entity"]
            label_cypher = ":" + ":".join(labels)
            
            # Remove connections from properties
            props = {k: v for k, v in node.items() if k not in ("connections",)}
            
            # Sanitize properties (remove None values, convert lists to strings if needed)
            props = self._sanitize_properties(props)
            
            query = (
                f"MERGE (n{label_cypher} {{id: $id}}) "
                "SET n += $props "
                "RETURN n.id as id"
            )
            
            result = tx.run(query, id=node["id"], props=props)
            record = result.single()
            
            if record:
                logger.debug(f"Node upserted: {record['id']}")
            
        except Exception as e:
            logger.error(f"Error in _upsert_node transaction: {e}")
            raise
    
    def _sanitize_properties(self, props: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize properties for Neo4j.
        
        Args:
            props: Properties dictionary
            
        Returns:
            Sanitized properties
        """
        sanitized = {}
        for key, value in props.items():
            if value is None:
                logger.debug(f"Skipping None value for property: {key}")
                continue
            
            # Convert complex types to strings
            if isinstance(value, (dict, list, tuple)):
                sanitized[key] = json.dumps(value) if value else ""
            else:
                sanitized[key] = value
        
        return sanitized
    
    def _create_relationship_with_retry(
        self, 
        session, 
        source_id: str, 
        rel: Dict[str, str]
    ) -> bool:
        """
        Create relationship with retry logic.
        
        Args:
            session: Neo4j session
            source_id: Source node ID
            rel: Relationship dictionary with 'relation' and 'target'
            
        Returns:
            True if successful, False otherwise
        """
        for attempt in range(self.MAX_RETRIES):
            try:
                session.execute_write(self._create_relationship, source_id, rel)
                logger.debug(f"Created relationship: {source_id} -> {rel.get('target')}")
                return True
            except TransientError as e:
                logger.warning(
                    f"Transient error on attempt {attempt + 1} for relationship "
                    f"{source_id} -> {rel.get('target')}: {e}"
                )
                if attempt == self.MAX_RETRIES - 1:
                    logger.error(
                        f"Failed to create relationship after {self.MAX_RETRIES} attempts"
                    )
                    return False
            except ClientError as e:
                logger.error(f"Client error creating relationship: {e}")
                return False
            except Exception as e:
                logger.error(f"Unexpected error creating relationship: {e}")
                return False
        
        return False
    
    def _create_relationship(
        self, 
        tx, 
        source_id: str, 
        rel: Dict[str, str]
    ) -> None:
        """
        Create a relationship between two nodes.
        
        Args:
            tx: Neo4j transaction
            source_id: Source node ID
            rel: Relationship dictionary with 'relation' and 'target'
        """
        try:
            rel_type = rel.get("relation", "RELATED_TO")
            target_id = rel.get("target")
            
            if not target_id:
                logger.debug(f"Skipping relationship with no target from {source_id}")
                return
            
            # Sanitize relationship type (Neo4j doesn't allow certain characters)
            rel_type = self._sanitize_relationship_type(rel_type)
            
            query = (
                "MATCH (a:Entity {id: $source_id}), (b:Entity {id: $target_id}) "
                f"MERGE (a)-[r:{rel_type}]->(b) "
                "RETURN type(r) as rel_type"
            )
            
            result = tx.run(query, source_id=source_id, target_id=target_id)
            record = result.single()
            
            if not record:
                logger.warning(
                    f"Could not create relationship: {source_id} -> {target_id}. "
                    "One or both nodes may not exist."
                )
            
        except Exception as e:
            logger.error(f"Error in _create_relationship transaction: {e}")
            raise
    
    def _sanitize_relationship_type(self, rel_type: str) -> str:
        """
        Sanitize relationship type for Neo4j.
        
        Args:
            rel_type: Original relationship type
            
        Returns:
            Sanitized relationship type
        """
        # Replace invalid characters with underscores
        sanitized = rel_type.replace(" ", "_").replace("-", "_")
        sanitized = "".join(c if c.isalnum() or c == "_" else "_" for c in sanitized)
        
        if sanitized != rel_type:
            logger.debug(f"Sanitized relationship type: {rel_type} -> {sanitized}")
        
        return sanitized
    
    def _create_constraints(self, tx) -> None:
        """
        Create database constraints.
        
        Args:
            tx: Neo4j transaction
        """
        try:
            query = "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE"
            tx.run(query)
            logger.info("Database constraints created successfully")
        except Exception as e:
            logger.error(f"Error creating constraints: {e}")
            raise
    
    def query_nodes(
        self, 
        node_type: Optional[str] = None, 
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Query nodes from the database.
        
        Args:
            node_type: Filter by node type (optional)
            limit: Maximum number of results
            
        Returns:
            List of node dictionaries
        """
        try:
            logger.info(f"Querying nodes (type={node_type}, limit={limit})")
            
            with self.driver.session() as session:
                if node_type:
                    query = f"MATCH (n:{node_type}) RETURN n LIMIT $limit"
                else:
                    query = "MATCH (n:Entity) RETURN n LIMIT $limit"
                
                result = session.run(query, limit=limit)
                nodes = [dict(record["n"]) for record in result]
                
                logger.info(f"Retrieved {len(nodes)} nodes")
                return nodes
                
        except Exception as e:
            logger.error(f"Error querying nodes: {e}")
            raise
    
    def get_statistics(self) -> Dict[str, int]:
        """
        Get database statistics.
        
        Returns:
            Dictionary with node and relationship counts
        """
        try:
            logger.info("Retrieving database statistics")
            
            with self.driver.session() as session:
                # Count nodes
                node_result = session.run("MATCH (n) RETURN count(n) as count")
                node_count = node_result.single()["count"]
                
                # Count relationships
                rel_result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
                rel_count = rel_result.single()["count"]
                
                stats = {
                    "nodes": node_count,
                    "relationships": rel_count
                }
                
                logger.info(f"Database statistics: {stats}")
                return stats
                
        except Exception as e:
            logger.error(f"Error retrieving statistics: {e}")
            raise


# Example usage
if __name__ == "__main__":
    try:
        with Neo4jClient() as client:
            client.upload_data()
            stats = client.get_statistics()
            print(f"\nDatabase Statistics: {stats}")
    except Exception as e:
        logger.error(f"Application error: {e}")
        raise

