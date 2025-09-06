from xai_sdk import Client
from xai_sdk.chat import user, system
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import re

class EvaluationDimension(Enum):
    ACCURACY = "accuracy"
    HELPFULNESS = "helpfulness"
    CLARITY = "clarity"
    COMPLETENESS = "completeness"
    SAFETY = "safety"
    CREATIVITY = "creativity"
    REASONING = "reasoning"

@dataclass
class EvaluationPrompt:
    category: str
    prompt: str
    reference_answer: Optional[str] = None
    evaluation_criteria: Optional[Dict[str, str]] = None
    difficulty: str = "medium"  # easy, medium, hard

class EnhancedGrokEvaluator:
    def __init__(self, api_key: str):
        self.client = Client(api_key=api_key)
        self.judge_model = "grok-4-0709"
        
    def get_enhanced_test_prompts(self) -> List[EvaluationPrompt]:
        """Get comprehensive test prompts with reference answers and criteria"""
        return [
            EvaluationPrompt(
                category="factual_accuracy",
                prompt="What year did the Apollo 11 moon landing occur, and who were the astronauts?",
                reference_answer="Apollo 11 landed on the moon in 1969. The astronauts were Neil Armstrong (first to walk), Buzz Aldrin (second to walk), and Michael Collins (command module pilot).",
                evaluation_criteria={
                    "accuracy": "Must mention 1969 and all three astronauts",
                    "completeness": "Should identify roles of astronauts"
                },
                difficulty="easy"
            ),
            EvaluationPrompt(
                category="math_reasoning",
                prompt="A store offers a 20% discount on a $150 item, then adds 8% sales tax. What's the final price?",
                reference_answer="$150 × 0.80 = $120 (after discount), $120 × 1.08 = $129.60 (with tax)",
                evaluation_criteria={
                    "accuracy": "Must arrive at $129.60",
                    "reasoning": "Should show calculation steps",
                    "clarity": "Steps should be easy to follow"
                },
                difficulty="medium"
            ),
            EvaluationPrompt(
                category="coding",
                prompt="Write a Python function to check if a string is a palindrome, ignoring spaces and case.",
                reference_answer="""def is_palindrome(s):
    cleaned = ''.join(s.lower().split())
    return cleaned == cleaned[::-1]""",
                evaluation_criteria={
                    "accuracy": "Function must correctly identify palindromes",
                    "completeness": "Must handle spaces and case insensitivity",
                    "clarity": "Code should be readable and efficient"
                },
                difficulty="medium"
            ),
            EvaluationPrompt(
                category="creative_writing",
                prompt="Write a haiku about artificial intelligence that includes a nature metaphor.",
                evaluation_criteria={
                    "creativity": "Should use original imagery",
                    "completeness": "Must follow 5-7-5 syllable structure",
                    "helpfulness": "Should include nature metaphor as requested"
                },
                difficulty="medium"
            ),
            EvaluationPrompt(
                category="ethical_reasoning",
                prompt="Should AI systems be required to identify themselves as non-human in all interactions? Provide arguments for both sides.",
                evaluation_criteria={
                    "completeness": "Must present both pro and con arguments",
                    "reasoning": "Arguments should be logical and well-structured",
                    "helpfulness": "Should consider practical implications"
                },
                difficulty="hard"
            ),
            EvaluationPrompt(
                category="complex_analysis",
                prompt="Explain how transformer architecture revolutionized NLP, including at least three specific innovations.",
                reference_answer="Key innovations: 1) Self-attention mechanism for capturing long-range dependencies, 2) Parallel processing vs sequential RNNs, 3) Positional encoding for sequence information",
                evaluation_criteria={
                    "accuracy": "Should mention self-attention, parallelization, and at least one other innovation",
                    "completeness": "Must explain why each innovation matters",
                    "clarity": "Technical concepts should be clearly explained"
                },
                difficulty="hard"
            )
        ]
    
    def comparative_judge(self, prompt: str, responses: Dict[str, str], 
                         criteria: Optional[Dict[str, str]] = None,
                         reference: Optional[str] = None) -> Dict[str, Any]:
        """Judge multiple responses comparatively"""
        
        judge_prompt = f"""You are an expert evaluator comparing AI model responses.

TASK: {prompt}

{'REFERENCE ANSWER: ' + reference if reference else ''}

{'EVALUATION CRITERIA:' if criteria else ''}
{json.dumps(criteria, indent=2) if criteria else ''}

RESPONSES TO EVALUATE:
{json.dumps(responses, indent=2)}

Provide a detailed evaluation following these steps:

1. ANALYSIS: Analyze each response against the criteria or general quality metrics
2. COMPARISON: Compare responses directly, noting strengths and weaknesses
3. SCORING: Score each response from 0.0 to 1.0 on relevant dimensions
4. RANKING: Rank the responses from best to worst
5. REASONING: Explain your scoring and ranking decisions

Output format:
```json
{{
  "analysis": {{
    "model_name": "detailed analysis text"
  }},
  "scores": {{
    "model_name": {{
      "overall": 0.0-1.0,
      "dimensions": {{
        "dimension_name": 0.0-1.0
      }}
    }}
  }},
  "ranking": ["best_model", "second_model", ...],
  "reasoning": "explanation of scoring decisions",
  "key_differences": "what distinguishes the responses"
}}
```"""

        try:
            chat = self.client.chat.create(
                model=self.judge_model,
                temperature=0.1  # Slightly higher for more nuanced evaluation
            )
            chat.append(system("You are an expert evaluator. Provide thorough, unbiased analysis."))
            chat.append(user(judge_prompt))
            
            result = chat.sample().content
            
            # Extract JSON with better error handling
            json_match = re.search(r'```json\s*(.*?)\s*```', result, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            else:
                # Try to find any JSON object
                json_match = re.search(r'\{.*\}', result, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
                    
        except Exception as e:
            print(f"Judge error: {e}")
            return self._default_scores(responses.keys())
    
    def pairwise_comparison(self, prompt: str, response_a: str, response_b: str,
                           model_a: str, model_b: str) -> Dict[str, Any]:
        """Direct pairwise comparison between two responses"""
        
        judge_prompt = f"""Compare these two AI responses to determine which is better.

TASK: {prompt}

RESPONSE A:
{response_a}

RESPONSE B:
{response_b}

Evaluate on these dimensions:
1. Accuracy/Correctness
2. Helpfulness/Completeness  
3. Clarity/Organization
4. Reasoning Quality (if applicable)

Provide your analysis in this format:
```json
{{
  "winner": "A" or "B" or "tie",
  "confidence": 0.0-1.0,
  "dimensions": {{
    "accuracy": {{"winner": "A/B/tie", "explanation": "..."}},
    "helpfulness": {{"winner": "A/B/tie", "explanation": "..."}},
    "clarity": {{"winner": "A/B/tie", "explanation": "..."}},
    "reasoning": {{"winner": "A/B/tie", "explanation": "..."}}
  }},
  "summary": "overall explanation"
}}
```"""

        try:
            chat = self.client.chat.create(
                model=self.judge_model,
                temperature=0
            )
            chat.append(system("You are an impartial judge. Evaluate based solely on response quality."))
            chat.append(user(judge_prompt))
            
            result = chat.sample().content
            json_match = re.search(r'```json\s*(.*?)\s*```', result, re.DOTALL)
            
            if json_match:
                comparison = json.loads(json_match.group(1))
                # Map back to model names
                if comparison.get("winner") == "A":
                    comparison["winner"] = model_a
                elif comparison.get("winner") == "B":
                    comparison["winner"] = model_b
                return comparison
                
        except Exception as e:
            print(f"Pairwise comparison error: {e}")
            
        return {"winner": "tie", "confidence": 0.5, "summary": "Evaluation failed"}
    
    def adversarial_evaluation(self, prompt: str, response: str) -> Dict[str, Any]:
        """Use adversarial prompting to find weaknesses"""
        
        adversarial_prompt = f"""You are a critical evaluator looking for potential issues in AI responses.

ORIGINAL TASK: {prompt}

RESPONSE TO EVALUATE:
{response}

Critically examine this response for:
1. Factual errors or inaccuracies
2. Logical inconsistencies or flawed reasoning
3. Missing important information
4. Potential biases or assumptions
5. Safety or ethical concerns
6. Unclear or confusing explanations

Provide a thorough critique:
```json
{{
  "issues_found": [
    {{"type": "category", "severity": "high/medium/low", "description": "..."}}
  ],
  "strengths": ["list of things done well"],
  "overall_quality": 0.0-1.0,
  "recommendation": "summary of how to improve"
}}
```"""

        try:
            chat = self.client.chat.create(
                model=self.judge_model,
                temperature=0.2
            )
            chat.append(system("You are a critical evaluator. Be thorough but fair."))
            chat.append(user(adversarial_prompt))
            
            result = chat.sample().content
            json_match = re.search(r'```json\s*(.*?)\s*```', result, re.DOTALL)
            
            if json_match:
                return json.loads(json_match.group(1))
                
        except Exception as e:
            print(f"Adversarial evaluation error: {e}")
            
        return {"issues_found": [], "overall_quality": 0.5}
    
    def consistency_check(self, model: str, prompt: str, num_samples: int = 3) -> Dict[str, Any]:
        """Check response consistency by sampling multiple times"""
        
        responses = []
        for _ in range(num_samples):
            chat = self.client.chat.create(
                model=model,
                temperature=0.7  # Some variation to test consistency
            )
            chat.append(user(prompt))
            response = chat.sample().content
            responses.append(response)
        
        # Analyze consistency
        consistency_prompt = f"""Analyze the consistency of these {num_samples} responses to the same prompt.

PROMPT: {prompt}

RESPONSES:
{json.dumps([f"Response {i+1}: {r}" for i, r in enumerate(responses)], indent=2)}

Evaluate:
1. Are the core facts/answers consistent?
2. Is the reasoning approach similar?
3. Are there any contradictions?
4. How much variation is there in quality?

Output:
```json
{{
  "consistency_score": 0.0-1.0,
  "factual_consistency": true/false,
  "approach_similarity": 0.0-1.0,
  "contradictions": ["list any contradictions"],
  "quality_variance": "low/medium/high",
  "analysis": "detailed explanation"
}}
```"""

        try:
            chat = self.client.chat.create(model=self.judge_model, temperature=0)
            chat.append(user(consistency_prompt))
            result = chat.sample().content
            
            json_match = re.search(r'```json\s*(.*?)\s*```', result, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
                
        except Exception as e:
            print(f"Consistency check error: {e}")
            
        return {"consistency_score": 0.5, "analysis": "Check failed"}
    
    def _default_scores(self, models):
        """Return default scores when evaluation fails"""
        return {
            "scores": {model: {"overall": 0.5} for model in models},
            "ranking": list(models),
            "reasoning": "Evaluation failed - default scores applied"
        }