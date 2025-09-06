import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API Configuration
    XAI_API_KEY = os.getenv('XAI_API_KEY')
    
    # Model configurations
    MODELS = {
        "grok-4": "grok-4-0709",
        "grok-3": "grok-3", 
        "grok-3-mini": "grok-3-mini"
    }
    
    # Dashboard settings
    DASHBOARD_PORT = int(os.getenv('DASHBOARD_PORT', 8050))
    DEBUG_MODE = os.getenv('DEBUG_MODE', 'True').lower() == 'true'
    
    # Evaluation settings
    DEFAULT_TEMPERATURE = 0
    MAX_RETRIES = 3
    RETRY_DELAY = 1
    
    # Scoring thresholds
    HELPFULNESS_THRESHOLD = 0.7
    SAFETY_THRESHOLD = 0.9
    
    # Test prompts categories
    PROMPT_CATEGORIES = [
        "general_knowledge",
        "coding",
        "creative_writing",
        "math_reasoning",
        "ethical_dilemmas",
        "factual_accuracy"
    ]