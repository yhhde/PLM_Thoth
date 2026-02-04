#7_generation_eval.py

"""
Generation evaluation script for bilingual Thoth model.

Generates text using beam search from trained model and saves outputs for LLM scoring.

Usage:
    python 7_generation_eval.py \
        --model_path /path/to/model/checkpoint.pt \
        --output_path ./generation_results.json \
        --device 0
"""

import os
import json
import argparse
import torch
from transformers import PreTrainedTokenizerFast
from model import GPT2


# --------------------------------------------------
# Test Prompts (50-100 bilingual prompts)
# --------------------------------------------------

TEST_PROMPTS = [
    # EN -> FR Translation style
    {"type": "en_to_fr", "prompt": "<en> Hello, how are you today? <fr>", "expected_lang": "fr"},
    {"type": "en_to_fr", "prompt": "<en> The weather is beautiful. <fr>", "expected_lang": "fr"},
    {"type": "en_to_fr", "prompt": "<en> I love learning new languages. <fr>", "expected_lang": "fr"},
    {"type": "en_to_fr", "prompt": "<en> What time is it? <fr>", "expected_lang": "fr"},
    {"type": "en_to_fr", "prompt": "<en> The book is on the table. <fr>", "expected_lang": "fr"},
    {"type": "en_to_fr", "prompt": "<en> She is reading a newspaper. <fr>", "expected_lang": "fr"},
    {"type": "en_to_fr", "prompt": "<en> The children are playing in the park. <fr>", "expected_lang": "fr"},
    {"type": "en_to_fr", "prompt": "<en> I would like a cup of coffee, please. <fr>", "expected_lang": "fr"},
    {"type": "en_to_fr", "prompt": "<en> The train arrives at noon. <fr>", "expected_lang": "fr"},
    {"type": "en_to_fr", "prompt": "<en> They are celebrating their anniversary. <fr>", "expected_lang": "fr"},
    {"type": "en_to_fr", "prompt": "<en> The museum opens at nine in the morning. <fr>", "expected_lang": "fr"},
    {"type": "en_to_fr", "prompt": "<en> My favorite color is blue. <fr>", "expected_lang": "fr"},
    {"type": "en_to_fr", "prompt": "<en> The doctor recommended more exercise. <fr>", "expected_lang": "fr"},
    {"type": "en_to_fr", "prompt": "<en> We need to buy groceries for dinner. <fr>", "expected_lang": "fr"},
    {"type": "en_to_fr", "prompt": "<en> The concert was absolutely amazing. <fr>", "expected_lang": "fr"},
    {"type": "en_to_fr", "prompt": "<en> Please pass me the salt. <fr>", "expected_lang": "fr"},
    {"type": "en_to_fr", "prompt": "<en> The flowers in the garden are blooming. <fr>", "expected_lang": "fr"},
    {"type": "en_to_fr", "prompt": "<en> I have an appointment tomorrow morning. <fr>", "expected_lang": "fr"},
    {"type": "en_to_fr", "prompt": "<en> The cat is sleeping on the sofa. <fr>", "expected_lang": "fr"},
    {"type": "en_to_fr", "prompt": "<en> Climate change is affecting the entire planet. <fr>", "expected_lang": "fr"},
    {"type": "en_to_fr", "prompt": "<en> The economy is growing steadily. <fr>", "expected_lang": "fr"},
    {"type": "en_to_fr", "prompt": "<en> Education is the key to success. <fr>", "expected_lang": "fr"},
    {"type": "en_to_fr", "prompt": "<en> Technology is changing our lives. <fr>", "expected_lang": "fr"},
    {"type": "en_to_fr", "prompt": "<en> Democracy requires active participation. <fr>", "expected_lang": "fr"},
    {"type": "en_to_fr", "prompt": "<en> The restaurant serves excellent food. <fr>", "expected_lang": "fr"},
    
    # FR -> EN Translation style
    {"type": "fr_to_en", "prompt": "<fr> Bonjour, comment allez-vous? <en>", "expected_lang": "en"},
    {"type": "fr_to_en", "prompt": "<fr> Il fait beau aujourd'hui. <en>", "expected_lang": "en"},
    {"type": "fr_to_en", "prompt": "<fr> J'aime beaucoup la musique classique. <en>", "expected_lang": "en"},
    {"type": "fr_to_en", "prompt": "<fr> Quelle heure est-il? <en>", "expected_lang": "en"},
    {"type": "fr_to_en", "prompt": "<fr> Le livre est sur la table. <en>", "expected_lang": "en"},
    {"type": "fr_to_en", "prompt": "<fr> Elle lit un journal. <en>", "expected_lang": "en"},
    {"type": "fr_to_en", "prompt": "<fr> Les enfants jouent dans le parc. <en>", "expected_lang": "en"},
    {"type": "fr_to_en", "prompt": "<fr> Je voudrais un café, s'il vous plaît. <en>", "expected_lang": "en"},
    {"type": "fr_to_en", "prompt": "<fr> Le train arrive à midi. <en>", "expected_lang": "en"},
    {"type": "fr_to_en", "prompt": "<fr> Ils fêtent leur anniversaire. <en>", "expected_lang": "en"},
    {"type": "fr_to_en", "prompt": "<fr> Le musée ouvre à neuf heures du matin. <en>", "expected_lang": "en"},
    {"type": "fr_to_en", "prompt": "<fr> Ma couleur préférée est le bleu. <en>", "expected_lang": "en"},
    {"type": "fr_to_en", "prompt": "<fr> Le médecin a recommandé plus d'exercice. <en>", "expected_lang": "en"},
    {"type": "fr_to_en", "prompt": "<fr> Nous devons acheter des provisions pour le dîner. <en>", "expected_lang": "en"},
    {"type": "fr_to_en", "prompt": "<fr> Le concert était absolument incroyable. <en>", "expected_lang": "en"},
    {"type": "fr_to_en", "prompt": "<fr> Passez-moi le sel, s'il vous plaît. <en>", "expected_lang": "en"},
    {"type": "fr_to_en", "prompt": "<fr> Les fleurs du jardin sont en pleine floraison. <en>", "expected_lang": "en"},
    {"type": "fr_to_en", "prompt": "<fr> J'ai un rendez-vous demain matin. <en>", "expected_lang": "en"},
    {"type": "fr_to_en", "prompt": "<fr> Le chat dort sur le canapé. <en>", "expected_lang": "en"},
    {"type": "fr_to_en", "prompt": "<fr> Le changement climatique affecte la planète entière. <en>", "expected_lang": "en"},
    {"type": "fr_to_en", "prompt": "<fr> L'économie croît régulièrement. <en>", "expected_lang": "en"},
    {"type": "fr_to_en", "prompt": "<fr> L'éducation est la clé du succès. <en>", "expected_lang": "en"},
    {"type": "fr_to_en", "prompt": "<fr> La technologie change nos vies. <en>", "expected_lang": "en"},
    {"type": "fr_to_en", "prompt": "<fr> La démocratie nécessite une participation active. <en>", "expected_lang": "en"},
    {"type": "fr_to_en", "prompt": "<fr> Le restaurant sert une excellente cuisine. <en>", "expected_lang": "en"},
    
    # Open-ended topic generation
    {"type": "topic_en", "prompt": "<en> Artificial intelligence is", "expected_lang": "en"},
    {"type": "topic_en", "prompt": "<en> The future of renewable energy", "expected_lang": "en"},
    {"type": "topic_en", "prompt": "<en> In today's world,", "expected_lang": "en"},
    {"type": "topic_en", "prompt": "<en> Scientists have discovered that", "expected_lang": "en"},
    {"type": "topic_en", "prompt": "<en> The most important thing in life is", "expected_lang": "en"},
    {"type": "topic_fr", "prompt": "<fr> L'intelligence artificielle est", "expected_lang": "fr"},
    {"type": "topic_fr", "prompt": "<fr> L'avenir des énergies renouvelables", "expected_lang": "fr"},
    {"type": "topic_fr", "prompt": "<fr> Dans le monde d'aujourd'hui,", "expected_lang": "fr"},
    {"type": "topic_fr", "prompt": "<fr> Les scientifiques ont découvert que", "expected_lang": "fr"},
    {"type": "topic_fr", "prompt": "<fr> La chose la plus importante dans la vie est", "expected_lang": "fr"},
    
    # Complex sentences
    {"type": "en_to_fr", "prompt": "<en> The international community must work together to address the challenges of climate change. <fr>", "expected_lang": "fr"},
    {"type": "en_to_fr", "prompt": "<en> Despite the economic difficulties, the company managed to increase its revenue. <fr>", "expected_lang": "fr"},
    {"type": "en_to_fr", "prompt": "<en> The new policy aims to promote sustainable development and environmental protection. <fr>", "expected_lang": "fr"},
    {"type": "fr_to_en", "prompt": "<fr> La communauté internationale doit travailler ensemble pour relever les défis du changement climatique. <en>", "expected_lang": "en"},
    {"type": "fr_to_en", "prompt": "<fr> Malgré les difficultés économiques, l'entreprise a réussi à augmenter son chiffre d'affaires. <en>", "expected_lang": "en"},
    {"type": "fr_to_en", "prompt": "<fr> La nouvelle politique vise à promouvoir le développement durable et la protection de l'environnement. <en>", "expected_lang": "en"},
    
    # Questions
    {"type": "en_to_fr", "prompt": "<en> What are the main causes of global warming? <fr>", "expected_lang": "fr"},
    {"type": "en_to_fr", "prompt": "<en> How can we improve our education system? <fr>", "expected_lang": "fr"},
    {"type": "en_to_fr", "prompt": "<en> Why is biodiversity important? <fr>", "expected_lang": "fr"},
    {"type": "fr_to_en", "prompt": "<fr> Quelles sont les principales causes du réchauffement climatique? <en>", "expected_lang": "en"},
    {"type": "fr_to_en", "prompt": "<fr> Comment pouvons-nous améliorer notre système éducatif? <en>", "expected_lang": "en"},
    {"type": "fr_to_en", "prompt": "<fr> Pourquoi la biodiversité est-elle importante? <en>", "expected_lang": "en"},
]


# --------------------------------------------------
# Beam Search Generation
# --------------------------------------------------

def beam_search_generate(
    model, 
    tokenizer, 
    input_ids, 
    attention_mask,
    max_new_tokens=50,
    num_beams=5,
    eos_token_id=None,
    pad_token_id=None,
    device='cuda'
):
    """
    Beam search generation for GPT-2 model.
    
    Args:
        model: GPT2 model
        tokenizer: tokenizer for decoding
        input_ids: [1, T] input token ids
        attention_mask: [1, T] attention mask
        max_new_tokens: maximum tokens to generate
        num_beams: number of beams for beam search
        eos_token_id: end of sequence token id
        pad_token_id: padding token id
        device: device to use
        
    Returns:
        generated_ids: [1, T+max_new_tokens] token ids
    """
    model.eval()
    
    if eos_token_id is None:
        eos_token_id = tokenizer.convert_tokens_to_ids("<eos>")
    if pad_token_id is None:
        pad_token_id = tokenizer.convert_tokens_to_ids("<pad>")
    
    # Initialize beams: (score, sequence, attention_mask)
    beams = [(0.0, input_ids.clone(), attention_mask.clone())]
    
    completed = []
    
    for step in range(max_new_tokens):
        all_candidates = []
        
        for score, seq, mask in beams:
            # Check if this beam is already complete
            if seq[0, -1].item() == eos_token_id:
                completed.append((score, seq, mask))
                continue
            
            # Truncate to max_seq_len if needed
            max_seq_len = model.config["model"]["max_seq_len"]
            if seq.size(1) > max_seq_len:
                seq = seq[:, -max_seq_len:]
                mask = mask[:, -max_seq_len:]
            
            # Forward pass
            with torch.no_grad():
                logits, _ = model(seq, attention_mask=mask, labels=None)
            
            # Get log probabilities for next token
            next_token_logits = logits[0, -1, :]
            log_probs = torch.log_softmax(next_token_logits, dim=-1)
            
            # Get top-k candidates
            top_log_probs, top_indices = torch.topk(log_probs, num_beams)
            
            for i in range(num_beams):
                new_token = top_indices[i].unsqueeze(0).unsqueeze(0)
                new_score = score + top_log_probs[i].item()
                new_seq = torch.cat([seq, new_token], dim=1)
                new_mask = torch.cat([mask, torch.ones(1, 1, device=device)], dim=1)
                
                all_candidates.append((new_score, new_seq, new_mask))
        
        if not all_candidates:
            break
            
        # Select top beams
        all_candidates.sort(key=lambda x: x[0], reverse=True)
        beams = all_candidates[:num_beams]
        
        # Early stopping if all beams completed
        if len(completed) >= num_beams:
            break
    
    # Add remaining beams to completed
    completed.extend(beams)
    
    # Return best sequence
    completed.sort(key=lambda x: x[0], reverse=True)
    best_seq = completed[0][1]
    
    return best_seq


# --------------------------------------------------
# Main Generation Loop
# --------------------------------------------------

def generate_all(model_path, tokenizer_path, output_path, device_idx=0, num_beams=5, max_new_tokens=50):
    """
    Generate text for all test prompts and save results.
    """
    device = torch.device(f"cuda:{device_idx}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load tokenizer
    print(f"Loading tokenizer from: {tokenizer_path}")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    
    # Load model
    print(f"Loading model from: {model_path}")
    model = GPT2.load(model_path).to(device)
    model.eval()
    print(f"Model loaded. Parameters: {model.get_num_params():,}")
    
    # Special token ids
    eos_id = tokenizer.convert_tokens_to_ids("<eos>")
    pad_id = tokenizer.convert_tokens_to_ids("<pad>")
    
    results = []
    
    print(f"\nGenerating {len(TEST_PROMPTS)} samples...")
    
    for i, prompt_data in enumerate(TEST_PROMPTS):
        prompt = prompt_data["prompt"]
        
        # Tokenize
        encoded = tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        
        # Generate
        output_ids = beam_search_generate(
            model=model,
            tokenizer=tokenizer,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            eos_token_id=eos_id,
            pad_token_id=pad_id,
            device=device
        )
        
        # Decode
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)
        generated_only = tokenizer.decode(output_ids[0, input_ids.size(1):], skip_special_tokens=True)
        
        result = {
            "id": i,
            "type": prompt_data["type"],
            "expected_lang": prompt_data["expected_lang"],
            "input_prompt": prompt,
            "full_output": generated_text,
            "generated_only": generated_only.strip(),
            "input_tokens": input_ids.size(1),
            "output_tokens": output_ids.size(1),
        }
        
        results.append(result)
        
        # Progress
        if (i + 1) % 10 == 0:
            print(f"  Generated {i+1}/{len(TEST_PROMPTS)}")
    
    # Save results
    print(f"\nSaving results to: {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_path": model_path,
            "num_prompts": len(TEST_PROMPTS),
            "num_beams": num_beams,
            "max_new_tokens": max_new_tokens,
            "results": results
        }, f, indent=2, ensure_ascii=False)
    
    print("Done!")
    
    # Print sample outputs
    print("\n" + "="*60)
    print("SAMPLE OUTPUTS")
    print("="*60)
    
    for r in results[:5]:
        print(f"\n[{r['type']}] Input: {r['input_prompt']}")
        print(f"Generated: {r['generated_only']}")


def main():
    parser = argparse.ArgumentParser(description="Generate text for evaluation")
    parser.add_argument("--model_path", required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--tokenizer_path", required=True, help="Path to tokenizer directory")
    parser.add_argument("--output_path", default="./generation_results.json", help="Output JSON path")
    parser.add_argument("--device", type=int, default=0, help="CUDA device")
    parser.add_argument("--num_beams", type=int, default=5, help="Number of beams for beam search")
    parser.add_argument("--max_new_tokens", type=int, default=50, help="Max tokens to generate")
    
    args = parser.parse_args()
    
    generate_all(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        output_path=args.output_path,
        device_idx=args.device,
        num_beams=args.num_beams,
        max_new_tokens=args.max_new_tokens
    )


if __name__ == "__main__":
    main()
