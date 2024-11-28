import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import threading

# Step 1: Load Models
def load_model(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

# Load models once at the start
model_gpt2, tokenizer_gpt2 = load_model("meta-llama/Meta-Llama-3-8B")
model_neo, tokenizer_neo = load_model("nomic-ai/gpt4all-j")

# Step 2: Predefined Expected Facts
expected_facts = {
    "history": ["World War I", "Enlightenment", "Cold War", "Silk Road", "women in leadership"]
}

# Step 3: Generate Responses (Reused for both models)
def generate_response(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=512, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Step 4: Parallel Hallucination Detection Functions

# Factual Hallucination Check (lightweight string matching)
def check_factual(response, expected_facts):
    missing_facts = [fact for fact in expected_facts if fact not in response]
    return len(missing_facts)

# Logical Hallucination Check (simple trigger words)
def check_logical(response):
    contradictions = ["but", "however", "although", "yet"]
    return any(word in response for word in contradictions)

# Contradictory Hallucination Check (Negation detection)
def check_contradictory(response):
    negations = ["not", "no", "never"]
    midpoint = len(response) // 2
    first_half = response[:midpoint]
    second_half = response[midpoint:]
    return any(neg in first_half and neg in second_half for neg in negations)

# Step 5: Run Parallel Hallucination Detection
def detect_hallucinations(response, expected_facts):
    results = {}
    
    # Parallel execution for each check
    threads = []
    results['factual'] = 0
    results['logical'] = 0
    results['contradictory'] = 0
    
    t1 = threading.Thread(target=lambda: results.update({"factual": check_factual(response, expected_facts)}))
    t2 = threading.Thread(target=lambda: results.update({"logical": check_logical(response)}))
    t3 = threading.Thread(target=lambda: results.update({"contradictory": check_contradictory(response)}))
    
    # Start threads
    threads.extend([t1, t2, t3])
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    return results

# Step 6: Real-Time Detection Framework
def real_time_detection(prompt):
    # Generate responses
    response_gpt2 = generate_response(model_gpt2, tokenizer_gpt2, prompt)
    response_neo = generate_response(model_neo, tokenizer_neo, prompt)

    # Detect hallucinations in parallel
    hallucination_gpt2 = detect_hallucinations(response_gpt2, expected_facts['history'])
    hallucination_neo = detect_hallucinations(response_neo, expected_facts['history'])

    # Flag responses based on any hallucination counts > 0
    flags = {
        "GPT-2": any(hallucination_gpt2.values()),
        "Neo": any(hallucination_neo.values())
    }

    return {
        "responses": {
            "GPT-2": response_gpt2,
            "Neo": response_neo
        },
        "hallucinations": {
            "GPT-2": hallucination_gpt2,
            "Neo": hallucination_neo
        },
        "flags": flags
    }

# Step 7: Example Usage
if __name__ == "__main__":
    prompt = "Explain the consequences of World War I on empires."
    detection_result = real_time_detection(prompt)
    
    print("Responses:")
    for model, response in detection_result["responses"].items():
        print(f"{model}: {response}")

    print("\nHallucination Counts:")
    for model, counts in detection_result["hallucinations"].items():
        print(f"{model}: {counts}")

    print("\nFlags:")
    for model, flag in detection_result["flags"].items():
        print(f"{model}: {'Hallucination Detected' if flag else 'No Hallucination'}")
