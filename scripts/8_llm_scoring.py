#8_llm_scoring.py

"""
LLM-based scoring script for generated bilingual text.

Supports multiple LLM models for scoring comparison.

Usage:
    python 8_llm_scoring.py \
        --input_path ./generation_results.json \
        --output_path ./scoring_results.json \
        --scorer mistral \
        --device 0
"""

import os
import json
import argparse
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# --------------------------------------------------
# Available Scorer Models
# --------------------------------------------------

SCORER_MODELS = {
    "mistral": {
        "name": "mistralai/Mistral-7B-Instruct-v0.2",
        "description": "Mistral 7B - French company, excellent multilingual",
        "memory_gb": 14
    },
    "croissant": {
        "name": "croissantllm/CroissantLLMChat-v0.1",
        "description": "CroissantLLM - Designed for EN-FR bilingual",
        "memory_gb": 14
    },
    "qwen": {
        "name": "Qwen/Qwen2.5-7B-Instruct",
        "description": "Qwen 2.5 - Strong multilingual capabilities",
        "memory_gb": 14
    },
    "llama": {
        "name": "meta-llama/Llama-3.1-8B-Instruct",
        "description": "Llama 3.1 - Meta's latest open model",
        "memory_gb": 16
    },
    "deepseek": {
        "name": "deepseek-ai/deepseek-llm-7b-chat",
        "description": "DeepSeek 7B - Good general performance",
        "memory_gb": 14
    },
    "phi": {
        "name": "microsoft/Phi-3-mini-4k-instruct",
        "description": "Phi-3 Mini - Small but capable (3.8B)",
        "memory_gb": 8
    }
}


# --------------------------------------------------
# Scoring Prompt Template
# --------------------------------------------------

SCORING_PROMPT_TEMPLATE = """You are an expert in evaluating bilingual English-French language models.

**Task Type:** {task_type}
**Input prompt given to the model:**
{input_prompt}

**Model's generated text:**
{generated_text}

Please evaluate the generated text on the following criteria (1-5 scale, where 1=very poor, 5=excellent):

1. **Fluency**: Is the text grammatically correct and natural sounding?
2. **Coherence**: Does the text make logical sense and flow well?
3. **Accuracy**: For translation tasks, is the meaning correctly conveyed? For open-ended tasks, is the content factually reasonable?
4. **Relevance**: Is the output appropriate and relevant to the input prompt?

Respond ONLY with a JSON object in this exact format:
{{"fluency": X, "coherence": X, "accuracy": X, "relevance": X, "overall": X, "comment": "brief explanation"}}

Your evaluation:"""


# --------------------------------------------------
# LLM Scorer Class
# --------------------------------------------------

class LLMScorer:
    """Scorer using various LLM models."""
    
    def __init__(self, scorer_key="mistral", device="cuda:0", use_4bit=False):
        """
        Initialize LLM model.
        
        Args:
            scorer_key: key from SCORER_MODELS
            device: device to use
            use_4bit: whether to use 4-bit quantization (saves memory)
        """
        if scorer_key not in SCORER_MODELS:
            raise ValueError(f"Unknown scorer: {scorer_key}. Available: {list(SCORER_MODELS.keys())}")
        
        model_info = SCORER_MODELS[scorer_key]
        model_name = model_info["name"]
        
        print(f"Loading scorer: {scorer_key}")
        print(f"  Model: {model_name}")
        print(f"  Description: {model_info['description']}")
        print(f"  Expected memory: ~{model_info['memory_gb']}GB")
        
        self.device = device
        self.scorer_key = scorer_key
        self.model_name = model_name
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Load model
        if use_4bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map=device,
                trust_remote_code=True
            )
            print("  Loaded with 4-bit quantization")
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map=device,
                trust_remote_code=True
            )
        
        self.model.eval()
        print(f"  Model loaded successfully!")
    
    def score(self, task_type, input_prompt, generated_text, max_new_tokens=256):
        """
        Score a single generation.
        
        Args:
            task_type: type of task (en_to_fr, fr_to_en, topic_en, topic_fr)
            input_prompt: the input prompt given to the model
            generated_text: the generated text to evaluate
            max_new_tokens: max tokens for scoring response
            
        Returns:
            dict with scores and comment
        """
        prompt = SCORING_PROMPT_TEMPLATE.format(
            task_type=task_type,
            input_prompt=input_prompt,
            generated_text=generated_text
        )
        
        # Handle chat format for instruction models
        if hasattr(self.tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            formatted = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            inputs = self.tokenizer(formatted, return_tensors="pt").to(self.device)
        else:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], 
            skip_special_tokens=True
        )
        
        # Parse JSON response
        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                scores = json.loads(json_str)
                return scores
            else:
                return {"error": "No JSON found", "raw_response": response}
        except json.JSONDecodeError as e:
            return {"error": str(e), "raw_response": response}


# --------------------------------------------------
# Main Scoring Loop
# --------------------------------------------------

def score_all(input_path, output_path, scorer_key, device_idx=0, use_4bit=False):
    """
    Score all generated texts and save results.
    """
    device = f"cuda:{device_idx}"
    
    # Load generation results
    print(f"Loading generations from: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    results = data["results"]
    print(f"Loaded {len(results)} generations")
    
    # Initialize scorer
    scorer = LLMScorer(scorer_key=scorer_key, device=device, use_4bit=use_4bit)
    
    # Score all
    scored_results = []
    
    for item in tqdm(results, desc=f"Scoring with {scorer_key}"):
        scores = scorer.score(
            task_type=item["type"],
            input_prompt=item["input_prompt"],
            generated_text=item["generated_only"]
        )
        
        scored_item = {
            **item,
            "scores": scores
        }
        scored_results.append(scored_item)
    
    # Calculate aggregate statistics
    valid_scores = [r["scores"] for r in scored_results if "error" not in r["scores"]]
    
    if valid_scores:
        avg_fluency = sum(s.get("fluency", 0) for s in valid_scores) / len(valid_scores)
        avg_coherence = sum(s.get("coherence", 0) for s in valid_scores) / len(valid_scores)
        avg_accuracy = sum(s.get("accuracy", 0) for s in valid_scores) / len(valid_scores)
        avg_relevance = sum(s.get("relevance", 0) for s in valid_scores) / len(valid_scores)
        avg_overall = sum(s.get("overall", 0) for s in valid_scores) / len(valid_scores)
        
        # By task type
        by_type = {}
        for r in scored_results:
            if "error" not in r["scores"]:
                t = r["type"]
                if t not in by_type:
                    by_type[t] = []
                by_type[t].append(r["scores"])
        
        type_averages = {}
        for t, scores in by_type.items():
            type_averages[t] = {
                "count": len(scores),
                "avg_fluency": sum(s.get("fluency", 0) for s in scores) / len(scores),
                "avg_coherence": sum(s.get("coherence", 0) for s in scores) / len(scores),
                "avg_accuracy": sum(s.get("accuracy", 0) for s in scores) / len(scores),
                "avg_relevance": sum(s.get("relevance", 0) for s in scores) / len(scores),
                "avg_overall": sum(s.get("overall", 0) for s in scores) / len(scores),
            }
        
        summary = {
            "total_samples": len(results),
            "valid_scores": len(valid_scores),
            "error_count": len(results) - len(valid_scores),
            "overall_averages": {
                "fluency": round(avg_fluency, 2),
                "coherence": round(avg_coherence, 2),
                "accuracy": round(avg_accuracy, 2),
                "relevance": round(avg_relevance, 2),
                "overall": round(avg_overall, 2),
            },
            "by_task_type": type_averages
        }
    else:
        summary = {
            "total_samples": len(results),
            "valid_scores": 0,
            "error_count": len(results),
            "error": "No valid scores obtained"
        }
    
    # Save results
    output_data = {
        "model_path": data["model_path"],
        "scorer_key": scorer_key,
        "scorer_model": SCORER_MODELS[scorer_key]["name"],
        "summary": summary,
        "results": scored_results
    }
    
    print(f"\nSaving results to: {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("\n" + "="*60)
    print(f"SCORING SUMMARY ({scorer_key})")
    print("="*60)
    
    if valid_scores:
        print(f"\nOverall Averages (1-5 scale):")
        print(f"  Fluency:   {avg_fluency:.2f}")
        print(f"  Coherence: {avg_coherence:.2f}")
        print(f"  Accuracy:  {avg_accuracy:.2f}")
        print(f"  Relevance: {avg_relevance:.2f}")
        print(f"  Overall:   {avg_overall:.2f}")
        
        print(f"\nBy Task Type:")
        for t, avgs in type_averages.items():
            print(f"  {t}: overall={avgs['avg_overall']:.2f} (n={avgs['count']})")
    else:
        print("No valid scores obtained!")
    
    print("\nDone!")


def list_scorers():
    """Print available scorer models."""
    print("\n" + "="*60)
    print("AVAILABLE SCORER MODELS")
    print("="*60)
    for key, info in SCORER_MODELS.items():
        print(f"\n  {key}:")
        print(f"    Model: {info['name']}")
        print(f"    Description: {info['description']}")
        print(f"    Memory: ~{info['memory_gb']}GB")


def main():
    parser = argparse.ArgumentParser(description="Score generated text with LLM")
    parser.add_argument("--input_path", help="Path to generation results JSON")
    parser.add_argument("--output_path", help="Output JSON path")
    parser.add_argument("--scorer", default="mistral", 
                        choices=list(SCORER_MODELS.keys()),
                        help="Scorer model to use")
    parser.add_argument("--device", type=int, default=0, help="CUDA device")
    parser.add_argument("--use_4bit", action="store_true", 
                        help="Use 4-bit quantization to save memory")
    parser.add_argument("--list_scorers", action="store_true",
                        help="List available scorer models and exit")
    
    args = parser.parse_args()
    
    if args.list_scorers:
        list_scorers()
        return
    
    if not args.input_path:
        parser.error("--input_path is required unless using --list_scorers")
    
    # Auto-generate output path if not provided
    if not args.output_path:
        base = os.path.splitext(args.input_path)[0]
        args.output_path = f"{base}_scored_{args.scorer}.json"
    
    score_all(
        input_path=args.input_path,
        output_path=args.output_path,
        scorer_key=args.scorer,
        device_idx=args.device,
        use_4bit=args.use_4bit
    )


if __name__ == "__main__":
    main()
