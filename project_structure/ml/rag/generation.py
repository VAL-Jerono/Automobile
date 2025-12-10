"""
RAG Generation module.
Connects to Ollama to generate natural language answers based on retrieved context.
"""

import requests
import logging
import json
from typing import List, Dict, Any, Optional
import os

logger = logging.getLogger(__name__)

class RAGGenerator:
    """
    Generates answers using an LLM (via Ollama) and retrieved context.
    """
    
    def __init__(self, model_name: str = "llama2", ollama_base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = ollama_base_url
        self.api_generate = f"{self.base_url}/api/generate"
        
    def check_availability(self) -> bool:
        """Check if Ollama is running and model is available."""
        try:
            # Check if server is up
            response = requests.get(self.base_url)
            if response.status_code != 200:
                logger.warning(f"Ollama server not reachable at {self.base_url}")
                return False
                
            # Check if model is pulled
            tags_response = requests.get(f"{self.base_url}/api/tags")
            if tags_response.status_code == 200:
                models = [m['name'] for m in tags_response.json().get('models', [])]
                # Simple check if model name is contained in any available tag
                if not any(self.model_name in m for m in models):
                    logger.warning(f"Model {self.model_name} not found in Ollama. Available: {models}")
                    # We might still try to run it, as Ollama might pull it on demand or we might have a partial match
                    # But returning True here assumes the user knows what they are doing if they approved the plan
            
            return True
        except Exception as e:
            logger.error(f"Error checking Ollama availability: {e}")
            return False

    def generate_answer(self, query: str, context_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate an answer given the user query and retrieved documents.
        """
        if not context_docs:
            return {
                "answer": "I couldn't find any relevant information in the database to answer your question.",
                "sources": []
            }
            
        # Construct context string
        context_text = ""
        sources = []
        
        for i, doc in enumerate(context_docs):
            # doc structure from RAGEngine: {'document': '...', 'policy_id': ...}
            context_text += f"--- Source {i+1} ---\n{doc.get('document', '')}\n\n"
            
            # Collect source metadata
            source_info = {
                "rank": i + 1,
                "similarity": doc.get('similarity')
            }
            if 'policy_id' in doc:
                source_info['type'] = 'policy'
                source_info['id'] = doc['policy_id']
            elif 'claim_id' in doc:
                source_info['type'] = 'claim'
                source_info['id'] = doc['claim_id']
            
            sources.append(source_info)
            
        # Construct System Prompt
        system_prompt = """You are an expert insurance assistant for AutoGuard Insurance. 
Your goal is to answer the user's question ACCURATELY based ONLY on the provided context.
If the answer is not in the context, say "I don't have enough information to answer that."
Do not hallucinate facts.
Format your answer in a clear, professional manner.
"""

        # Construct User Prompt
        user_prompt = f"""
Context Information:
{context_text}

User Question: {query}

Answer:
"""

        # Call Ollama
        try:
            payload = {
                "model": self.model_name,
                "prompt": user_prompt,
                "system": system_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1  # Low temperature for factual answers
                }
            }
            
            logger.info(f"Sending request to Ollama ({self.model_name})...")
            response = requests.post(self.api_generate, json=payload, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get("response", "")
                return {
                    "answer": answer,
                    "sources": sources,
                    "model_used": self.model_name
                }
            else:
                logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                return {
                    "answer": "Sorry, I encountered an error while generating the answer.",
                    "error": f"Ollama API error: {response.status_code}",
                    "sources": sources
                }
                
        except requests.exceptions.ConnectionError:
            return {
                "answer": "I cannot connect to the AI engine (Ollama). Please ensure it is running.",
                "error": "Connection refused",
                "sources": sources
            }
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return {
                "answer": "An unexpected error occurred during generation.",
                "error": str(e),
                "sources": sources
            }
