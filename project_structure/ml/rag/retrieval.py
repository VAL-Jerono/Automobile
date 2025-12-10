"""
RAG (Retrieval-Augmented Generation) system for insurance policies and claims.
Uses ChromaDB + sentence-transformers for semantic search.
"""

import pandas as pd
import numpy as np
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("⚠️  sentence-transformers not available, RAG features disabled")
    
try:
    import chromadb
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    print("⚠️  ChromaDB not available, RAG features disabled")
    
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class RAGEngine:
    """
    RAG system for retrieving relevant policies and claims context.
    """
    
    def __init__(self, embedding_model: str = 'sentence-transformers/all-MiniLM-L6-v2',
                 vector_db_path: str = './vector_db', similarity_threshold: float = 0.2):
        if not SENTENCE_TRANSFORMERS_AVAILABLE or not CHROMADB_AVAILABLE:
            logger.warning("RAG dependencies not available - RAGEngine will not function")
            self.embedding_model = None
            self.client = None
            self.policy_collection = None
            self.claims_collection = None
            return
            
        self.embedding_model = SentenceTransformer(embedding_model)
        self.vector_db_path = vector_db_path
        self.similarity_threshold = similarity_threshold
        
        # Initialize ChromaDB
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=vector_db_path)
        
        # Try to load existing collections
        try:
            self.policy_collection = self.client.get_collection(name="policies")
            logger.info("Loaded existing 'policies' collection")
        except Exception:
            self.policy_collection = None
            
        try:
            self.claims_collection = self.client.get_collection(name="claims")
            logger.info("Loaded existing 'claims' collection")
        except Exception:
            self.claims_collection = None
    
    def index_policies(self, df_policies: pd.DataFrame):
        """Index policy documents in vector DB."""
        logger.info(f"Indexing {len(df_policies)} policies...")
        
        # Create policy collection
        self.policy_collection = self.client.get_or_create_collection(
            name="policies",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Process policies into documents
        docs = []
        metadatas = []
        ids = []
        
        for _, row in df_policies.iterrows():
            doc_text = f"""
            Policy ID: {row['policy_id']}
            Customer Age: {row['customer_age']}
            Vehicle Type: {row['type_fuel']}, Age: {row['vehicle_age']} years
            Premium: ${row['premium']:.2f}
            Claims History: {row['n_claims_history']}
            Lapse Status: {'Yes' if row['lapse'] else 'No'}
            """
            
            docs.append(doc_text)
            metadatas.append({
                'policy_id': str(row['policy_id']),
                'premium': str(row['premium']),
                'lapse': str(row['lapse'])
            })
            ids.append(f"policy_{row['policy_id']}")
        
        # Embed and store
        embeddings = self.embedding_model.encode(docs, show_progress_bar=True)
        self.policy_collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=docs,
            metadatas=metadatas
        )
        
        logger.info(f"✓ Indexed {len(docs)} policies")
    
    def index_claims(self, df_claims: pd.DataFrame):
        """Index claim documents in vector DB."""
        logger.info(f"Indexing {len(df_claims)} claims...")
        
        # Create claims collection
        self.claims_collection = self.client.get_or_create_collection(
            name="claims",
            metadata={"hnsw:space": "cosine"}
        )
        
        docs = []
        metadatas = []
        ids = []
        
        for _, row in df_claims.iterrows():
            doc_text = f"""
            Claim ID: {row['claim_id']}
            Policy ID: {row['policy_id']}
            Claim Date: {row['claim_date']}
            Amount: ${row['claim_amount']:.2f}
            Type: {row['claim_type']}
            Status: {row['claim_status']}
            """
            
            docs.append(doc_text)
            metadatas.append({
                'claim_id': str(row['claim_id']),
                'policy_id': str(row['policy_id']),
                'amount': str(row['claim_amount']),
                'status': row['claim_status']
            })
            ids.append(f"claim_{row['claim_id']}")
        
        embeddings = self.embedding_model.encode(docs, show_progress_bar=True)
        self.claims_collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=docs,
            metadatas=metadatas
        )
        
        logger.info(f"✓ Indexed {len(docs)} claims")
    
    def query_policies(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve similar policies."""
        if self.policy_collection is None:
            logger.warning("Policy collection not indexed")
            return []
        
        query_embedding = self.embedding_model.encode(query_text)
        results = self.policy_collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            include=['distances', 'documents', 'metadatas']
        )
        
        # Format results
        output = []
        for i, (doc, metadata, dist) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        )):
            similarity = 1 - dist  # Convert distance to similarity
            if similarity >= self.similarity_threshold:
                output.append({
                    'rank': i + 1,
                    'policy_id': metadata.get('policy_id'),
                    'similarity': float(similarity),
                    'document': doc
                })
        
        return output
    
    def query_claims(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve similar claims."""
        if self.claims_collection is None:
            logger.warning("Claims collection not indexed")
            return []
        
        query_embedding = self.embedding_model.encode(query_text)
        results = self.claims_collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            include=['distances', 'documents', 'metadatas']
        )
        
        output = []
        for i, (doc, metadata, dist) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        )):
            similarity = 1 - dist
            if similarity >= self.similarity_threshold:
                output.append({
                    'rank': i + 1,
                    'claim_id': metadata.get('claim_id'),
                    'policy_id': metadata.get('policy_id'),
                    'similarity': float(similarity),
                    'document': doc
                })
        
        return output
    
    def persist(self):
        """Persist vector DB to disk."""
        # PersistentClient saves automatically
        logger.info("✓ Vector DB persisted")
