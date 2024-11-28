import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load Models
def load_model(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

# Load models
model_gpt2, tokenizer_gpt2 = load_model("meta-llama/Meta-Llama-3-8B")
model_neo, tokenizer_neo = load_model("nomic-ai/gpt4all-j")

# Generate Response
def generate_response(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=512, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Identify Hallucinations
def identify_hallucinations(response, expected_facts):
    hallucinations = {'factual': 0, 'logical': 0, 'contradictory': 0}

    # Factual Hallucination Check
    for fact in expected_facts:
        if fact not in response:
            hallucinations['factual'] += 1

    # Logical Hallucination Check (simplified)
    contradictions = ["but", "however", "although", "yet"]
    if any(word in response for word in contradictions):
        hallucinations['logical'] += 1

    # Contradictory Hallucination Check
    midpoint = len(response) // 2
    first_half = response[:midpoint]
    second_half = response[midpoint:]
    if any(neg in first_half and neg in second_half for neg in ["not", "no", "never"]):
        hallucinations['contradictory'] += 1

    return hallucinations

# Expected facts for the dystopia theme comparison
expected_facts = [
    "totalitarianism", "control", "freedom", "gender roles", "surveillance", 
    "propaganda", "individualism", "rebellion", "society", "oppression"
]

# Main Execution
def main():
    # Define the prompt here
    prompt = "Compare and contrast the themes of dystopia in '1984' by George Orwell and 'The Handmaid's Tale' by Margaret Atwood."

    # Generate responses for both models
    response_gpt2 = generate_response(model_gpt2, tokenizer_gpt2, prompt)
    response_neo = generate_response(model_neo, tokenizer_neo, prompt)

    # Perplexity AI response (manually input)
    perplexity_response = """Comparison of Dystopian Themes in "1984" and "The Handmaid's Tale"
    Both "1984" by George Orwell and "The Handmaid's Tale" by Margaret Atwood present chilling visions of dystopian societies, exploring themes of totalitarianism, surveillance, identity, and sexual repression...
    (complete the text as per your response)"""

    # Analyze hallucinations
    hallucination_gpt2 = identify_hallucinations(response_gpt2, expected_facts)
    hallucination_neo = identify_hallucinations(response_neo, expected_facts)
    hallucination_perplexity = identify_hallucinations(perplexity_response, expected_facts)

    # Print Hallucination Counts
    print("Hallucination Counts:")
    print(f"meta-llama/Meta-Llama-3-8B: {hallucination_gpt2}")
    print(f"nomic-ai/gpt4all-j: {hallucination_neo}")
    print(f"Perplexity AI: {hallucination_perplexity}")

    # Cosine Similarity Calculation
    documents = [response_gpt2, response_neo, perplexity_response]
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()
    cosine_similarities = cosine_similarity(vectors)

    # Print Cosine Similarity Matrix
    print("\nCosine Similarity Matrix:")
    print(cosine_similarities)

    # Save detailed results per model
    output_folder = "hallucination_analysis"
    os.makedirs(output_folder, exist_ok=True)

    results = {
        'meta-llama/Meta-Llama-3-8B': {
            "Response": response_gpt2,
            "Factual Hallucinations": hallucination_gpt2['factual'],
            "Logical Hallucinations": hallucination_gpt2['logical'],
            "Contradictory Hallucinations": hallucination_gpt2['contradictory']
        },
        'nomic-ai/gpt4all-j': {
            "Response": response_neo,
            "Factual Hallucinations": hallucination_neo['factual'],
            "Logical Hallucinations": hallucination_neo['logical'],
            "Contradictory Hallucinations": hallucination_neo['contradictory']
        },
        'Perplexity AI': {
            "Response": perplexity_response,
            "Factual Hallucinations": hallucination_perplexity['factual'],
            "Logical Hallucinations": hallucination_perplexity['logical'],
            "Contradictory Hallucinations": hallucination_perplexity['contradictory']
        }
    }

    # Save results to CSV
    for model_name, data in results.items():
        df = pd.DataFrame([{
            "Model": model_name,
            "Response": data["Response"],
            "Factual Hallucinations": data["Factual Hallucinations"],
            "Logical Hallucinations": data["Logical Hallucinations"],
            "Contradictory Hallucinations": data["Contradictory Hallucinations"]
        }])
        df.to_csv(f"{output_folder}/{model_name.replace('/', '_')}_hallucination_details.csv", index=False)

    print(f"\nResults saved to '{output_folder}' folder.")

if __name__ == "__main__":
    main()
