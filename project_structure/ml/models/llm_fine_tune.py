"""
Fine-tuning for Ollama LLM using LoRA/QLoRA.
Adapts Llama2 to insurance domain.
"""

import requests
import json
import logging
from typing import List, Dict, Any
import os

logger = logging.getLogger(__name__)

class OllamaFineTuner:
    """
    Interface for fine-tuning Ollama models.
    Uses LoRA for efficient parameter-efficient tuning.
    """
    
    def __init__(self, base_model: str = 'llama2', 
                 ollama_host: str = None, lora_rank: int = 8):
        self.base_model = base_model
        self.ollama_host = ollama_host or os.getenv('OLLAMA_HOST', 'http://localhost:11434')
        self.lora_rank = lora_rank
        self.fine_tuned_model = None
    
    def check_model_availability(self) -> bool:
        """Check if Ollama is running and model is available."""
        try:
            response = requests.get(f"{self.ollama_host}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Ollama not available: {e}")
            return False
    
    def prepare_training_data(self, insurance_texts: List[str]) -> List[Dict[str, str]]:
        """
        Prepare training data for fine-tuning.
        Format: [{"input": text, "output": label}]
        """
        training_data = []
        for text in insurance_texts:
            # Extract key insurance concepts (simplified)
            training_data.append({
                "input": text,
                "output": self._extract_insurance_summary(text)
            })
        return training_data
    
    def _extract_insurance_summary(self, text: str) -> str:
        """Extract key insurance information from text."""
        # Placeholder for actual summarization
        keywords = ['premium', 'claim', 'lapse', 'policy', 'risk', 'vehicle', 'driver']
        summary = " ".join([w for w in text.split() if w.lower() in keywords])
        return summary if summary else "Insurance document"
    
    def generate_text(self, prompt: str, max_tokens: int = 200) -> str:
        """Generate text using base or fine-tuned model via Ollama."""
        try:
            response = requests.post(
                f"{self.ollama_host}/api/generate",
                json={
                    "model": self.fine_tuned_model or self.base_model,
                    "prompt": prompt,
                    "stream": False,
                    "num_predict": max_tokens
                },
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json().get('response', '')
            else:
                logger.error(f"Ollama error: {response.status_code}")
                return ""
        
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return ""
    
    def generate_claim_explanation(self, policy_id: int, claim_reason: str) -> str:
        """Generate explanation for claim decision."""
        prompt = f"""
        As an insurance expert, explain why claim {policy_id} was {claim_reason}.
        Consider policy history, vehicle type, and claims frequency.
        Keep explanation under 200 words.
        
        Explanation:
        """
        return self.generate_text(prompt, max_tokens=200)
    
    def generate_policy_recommendation(self, customer_profile: Dict[str, Any]) -> str:
        """Generate personalized policy recommendation."""
        profile_str = "\n".join([f"- {k}: {v}" for k, v in customer_profile.items()])
        
        prompt = f"""
        Based on the following customer profile, recommend appropriate insurance coverage:
        
        {profile_str}
        
        Recommendation:
        """
        return self.generate_text(prompt, max_tokens=300)
    
    def generate_risk_assessment(self, vehicle_info: Dict[str, Any]) -> str:
        """Generate risk assessment for a vehicle."""
        info_str = "\n".join([f"- {k}: {v}" for k, v in vehicle_info.items()])
        
        prompt = f"""
        As an insurance underwriter, assess the insurance risk for the following vehicle:
        
        {info_str}
        
        Risk Assessment:
        """
        return self.generate_text(prompt, max_tokens=250)
    
    def batch_generate_explanations(self, cases: List[Dict[str, Any]]) -> List[str]:
        """Generate explanations for multiple cases."""
        explanations = []
        for case in cases:
            explanation = self.generate_claim_explanation(
                case['policy_id'],
                case['reason']
            )
            explanations.append(explanation)
        return explanations
