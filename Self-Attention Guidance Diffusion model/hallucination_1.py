import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict
import pandas as pd
import os

# Step 1: Load Models
def load_model(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

# Load models
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

# Step 4: Identify Hallucinations
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
    # Compare the first half and second half for contradictions (e.g., negations of previous statements)
    midpoint = len(response) // 2
    first_half = response[:midpoint]
    second_half = response[midpoint:]
    if any(neg in first_half and neg in second_half for neg in ["not", "no", "never"]):
        hallucinations['contradictory'] += 1

    return hallucinations

# Expected facts for each domain
expected_facts = {
    "history": [
        "Enlightenment", "World War I", "Silk Road", "Cold War", "women in leadership"
    ],
    "technology": [
        "internet", "AI ethics", "communication", "renewable energy", "quantum computing"
    ],
    "medicine": [
        "vaccination", "genomics", "antibiotic resistance", "mental health", "CRISPR"
    ]
}

# Step 5: Main Execution
def main():
    results = {model_gpt2.config._name_or_path: defaultdict(list), model_neo.config._name_or_path: defaultdict(list)}
    hallucination_counts = {model_gpt2.config._name_or_path: defaultdict(lambda: {'factual': 0, 'logical': 0, 'contradictory': 0}),
                            model_neo.config._name_or_path: defaultdict(lambda: {'factual': 0, 'logical': 0, 'contradictory': 0})}

    for domain, prompts_list in prompts.items():
        for prompt in prompts_list:
            response_gpt2 = generate_response(model_gpt2, tokenizer_gpt2, prompt)
            response_neo = generate_response(model_neo, tokenizer_neo, prompt)

            # Analyze hallucinations
            hallucination_gpt2 = identify_hallucinations(response_gpt2, prompt, expected_facts[domain])
            hallucination_neo = identify_hallucinations(response_neo, prompt, expected_facts[domain])

            # Update results
            results[model_gpt2.config._name_or_path][domain].append((prompt, response_gpt2, hallucination_gpt2))
            results[model_neo.config._name_or_path][domain].append((prompt, response_neo, hallucination_neo))

            # Aggregate hallucination counts per domain
            for key in hallucination_gpt2.keys():
                hallucination_counts[model_gpt2.config._name_or_path][domain][key] += hallucination_gpt2[key]
                hallucination_counts[model_neo.config._name_or_path][domain][key] += hallucination_neo[key]

    # Step 6: Output Results and Save to CSV
    output_folder = "hallucination_analysis"
    os.makedirs(output_folder, exist_ok=True)

    # Save detailed results per model and domain
    for model_name, domain_results in results.items():
        model_data = []
        for domain, responses in domain_results.items():
            for prompt, response, hallucination in responses:
                model_data.append({
                    "Model": model_name,
                    "Domain": domain,
                    "Prompt": prompt,
                    "Response": response,
                    "Factual Hallucinations": hallucination['factual'],
                    "Logical Hallucinations": hallucination['logical'],
                    "Contradictory Hallucinations": hallucination['contradictory']
                })
        df = pd.DataFrame(model_data)
        df.to_csv(f"{output_folder}/{model_name.replace('/', '_')}_hallucination_details.csv", index=False)

    # Save hallucination taxonomy summary
    summary_data = []
    for model_name, domain_counts in hallucination_counts.items():
        for domain, counts in domain_counts.items():
            total_hallucinations = sum(counts.values())
            summary_data.append({
                "Model": model_name,
                "Domain": domain,
                "Factual Hallucinations": counts['factual'],
                "Logical Hallucinations": counts['logical'],
                "Contradictory Hallucinations": counts['contradictory'],
                "Total Hallucinations": total_hallucinations
            })

    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(f"{output_folder}/hallucination_summary.csv", index=False)
    print(f"Results saved to '{output_folder}' folder.")

if __name__ == "__main__":
    main()
