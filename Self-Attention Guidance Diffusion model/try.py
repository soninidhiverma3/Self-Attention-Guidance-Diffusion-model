import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict
import pandas as pd
import os
import math

# Step 1: Load Models
def load_model(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

# Load models
model_mistral, tokenizer_mistral = load_model("mistralai/Mistral-7B-v0.1")
model_gpt2, tokenizer_gpt2 = load_model("meta-llama/Meta-Llama-3-8B")
model_neo, tokenizer_neo = load_model("nomic-ai/gpt4all-j")

# Step 2: Define Complex, Multi-part Prompts
prompts = {
    "history": [
        "Compare the causes and effects of the American and French Revolutions, focusing on economic, social, and philosophical factors. How did Enlightenment ideals shape both?",
        "Discuss how World War I led to the decline of empires, such as the Austro-Hungarian and Ottoman Empires. What role did colonialism play in this process?",
        "Analyze the influence of the Silk Road on cultural, technological, and religious exchanges between East and West. How did it evolve during the Tang and Mongol periods?",
        "Explain how the Cold War impacted global politics, economics, and technology, comparing the strategies of the US and USSR in the Space Race and nuclear arms development.",
        "Examine the role of women in leadership and military roles during ancient and medieval times. Compare this with women's involvement in leadership during World War II."
    ],
    "technology": [
        "Assess the impact of the internet on the development of global economies and cultures, comparing it to the impact of the printing press in the 16th century.",
        "Discuss the ethical implications of AI in decision-making across fields such as healthcare, law enforcement, and autonomous vehicles. How might biases in algorithms be addressed?",
        "Compare the evolution of communication technologies from the telegraph to 5G networks. How have these developments affected global information dissemination and geopolitics?",
        "Analyze how renewable energy technologies, like solar and wind power, are reshaping global energy markets. What challenges and opportunities do they present compared to fossil fuels?",
        "Evaluate the potential of quantum computing in cybersecurity. How does it differ from traditional encryption methods, and what are the risks and benefits?"
    ],
    "medicine": [
        "Compare the evolution of vaccine development from the discovery of smallpox vaccination to mRNA technology in COVID-19. What challenges have arisen in vaccine distribution?",
        "Discuss the role of genomics in personalized medicine. How does it differ from traditional treatments, and what ethical concerns does it raise?",
        "Analyze the impact of antibiotic resistance on global health. How have practices in medicine and agriculture contributed to this problem, and what solutions exist?",
        "Examine the relationship between mental health awareness and public health policies over the past 50 years. How have societal attitudes toward mental illness evolved?",
        "Evaluate the potential of CRISPR technology in treating genetic disorders. Compare its effectiveness with other gene therapies, and discuss ethical considerations."
    ]
}

# Step 3: Generate Responses
def generate_response(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=512, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Step 4: Calculate Perplexity
def calculate_perplexity(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        perplexity = math.exp(loss.item())
    return perplexity

# Step 5: Identify Hallucinations
def identify_hallucinations(response, prompt, expected_facts):
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

# Expected facts for each domain
expected_facts = {
    "history": ["Enlightenment", "World War I", "Silk Road", "Cold War", "women in leadership"],
    "technology": ["internet", "AI ethics", "communication", "renewable energy", "quantum computing"],
    "medicine": ["vaccination", "genomics", "antibiotic resistance", "mental health", "CRISPR"]
}

# Step 6: Main Execution
def main():
    output_folder = "try_analysis"
    os.makedirs(output_folder, exist_ok=True)

    results = defaultdict(list)
    perplexity_scores = defaultdict(list)
    hallucination_summary = defaultdict(lambda: defaultdict(lambda: {'factual': 0, 'logical': 0, 'contradictory': 0}))

    for domain, prompts_list in prompts.items():
        for prompt in prompts_list:
            # Generate responses from Neo and LLM
            for model_name, model, tokenizer in [("GPT-2", model_gpt2, tokenizer_gpt2),
                                                 ("GPT-Neo", model_neo, tokenizer_neo)]:
                response = generate_response(model, tokenizer, prompt)
                perplexity = calculate_perplexity(model, tokenizer, prompt)
                hallucinations = identify_hallucinations(response, prompt, expected_facts[domain])

                results[model_name].append({
                    "Domain": domain,
                    "Prompt": prompt,
                    "Response": response,
                    "Factual Hallucinations": hallucinations['factual'],
                    "Logical Hallucinations": hallucinations['logical'],
                    "Contradictory Hallucinations": hallucinations['contradictory']
                })

                perplexity_scores[model_name].append({
                    "Domain": domain,
                    "Prompt": prompt,
                    "Perplexity": perplexity
                })

                for halluc_type in hallucinations:
                    hallucination_summary[model_name][domain][halluc_type] += hallucinations[halluc_type]

            # Generate response from Perplexity-LLM (Mistral)
            perplexity_response = generate_response(model_mistral, tokenizer_mistral, prompt)
            perplexity_llm = calculate_perplexity(model_mistral, tokenizer_mistral, prompt)
            hallucinations_llm = identify_hallucinations(perplexity_response, prompt, expected_facts[domain])

            results["Perplexity-LLM"].append({
                "Domain": domain,
                "Prompt": prompt,
                "Response": perplexity_response,
                "Factual Hallucinations": hallucinations_llm['factual'],
                "Logical Hallucinations": hallucinations_llm['logical'],
                "Contradictory Hallucinations": hallucinations_llm['contradictory']
            })

            perplexity_scores["Perplexity-LLM"].append({
                "Domain": domain,
                "Prompt": prompt,
                "Perplexity": perplexity_llm
            })

            for halluc_type in hallucinations_llm:
                hallucination_summary["Perplexity-LLM"][domain][halluc_type] += hallucinations_llm[halluc_type]

    # Save detailed results per model
    for model_name, data in results.items():
        df = pd.DataFrame(data)
        df.to_csv(f"{output_folder}/{model_name}_detailed_responses.csv", index=False)

    # Save perplexity scores per model
    for model_name, data in perplexity_scores.items():
        df = pd.DataFrame(data)
        df.to_csv(f"{output_folder}/{model_name}_perplexity_scores.csv", index=False)

    # Save hallucination summaries per model
    summary_data = []
    for model_name, domains in hallucination_summary.items():
        for domain, counts in domains.items():
            summary_data.append({
                "Model": model_name,
                "Domain": domain,
                "Total Factual Hallucinations": counts['factual'],
                "Total Logical Hallucinations": counts['logical'],
                "Total Contradictory Hallucinations": counts['contradictory']
            })

    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(f"{output_folder}/hallucination_summary.csv", index=False)

    print(f"Detailed responses, perplexity scores, and hallucination summaries saved in '{output_folder}' folder.")

if __name__ == "__main__":
    main()
