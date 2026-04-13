"""
Passo 1: Engenharia de Dados Sintéticos
Gera 55 pares prompt/response no domínio de Ciência de Dados e Machine Learning
utilizando a API do Google Gemini (gemini-1.5-flash), e salva em formato .jsonl.
"""

import os
import json
import random
import google.generativeai as genai

# Configuração
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY_HERE")
DOMAIN = "Ciência de Dados e Machine Learning"
NUM_PAIRS = 55
TRAIN_RATIO = 0.90
TRAIN_FILE = "data/train.jsonl"
TEST_FILE  = "data/test.jsonl"

# Inicializa o cliente Gemini
genai.configure(api_key=GEMINI_API_KEY)
_model = genai.GenerativeModel("gemini-1.5-flash")

# Tópicos para variar as perguntas geradas
TOPICS = [
    "regressão linear e logística",
    "árvores de decisão e Random Forest",
    "redes neurais e deep learning",
    "métricas de avaliação de modelos (AUC, F1, RMSE)",
    "engenharia de features e pré-processamento",
    "overfitting, underfitting e regularização",
    "cross-validation e seleção de modelos",
    "clusterização (K-Means, DBSCAN)",
    "processamento de linguagem natural (NLP)",
    "visão computacional e CNNs",
]


def generate_pair(topic: str) -> dict:
    """Solicita ao Gemini um par instruction/response sobre o tópico dado."""
    prompt = (
        "Você é um especialista em Ciência de Dados. "
        "Gere UM único par de pergunta e resposta em português, no formato JSON, "
        "com as chaves 'instruction' e 'response'. "
        "A pergunta deve ser clara e específica; a resposta deve ser didática, "
        "com no mínimo 3 sentenças. "
        "Responda SOMENTE com o JSON puro, sem markdown, sem backticks, sem texto extra.\n\n"
        f"Tópico: {topic}"
    )

    response = _model.generate_content(prompt)
    raw = response.text.strip()

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    return json.loads(raw.strip())


def main():
    os.makedirs("data", exist_ok=True)

    print(f"[INFO] Gerando {NUM_PAIRS} pares de instruction/response...")
    pairs: list[dict] = []

    for i in range(NUM_PAIRS):
        topic = TOPICS[i % len(TOPICS)]
        try:
            pair = generate_pair(topic)
            if "instruction" in pair and "response" in pair:
                pairs.append(pair)
                print(f"  [{i+1:02d}/{NUM_PAIRS}] ✔  tópico: {topic[:40]}")
            else:
                print(f"  [{i+1:02d}/{NUM_PAIRS}] ✘  par inválido, pulando.")
        except Exception as exc:
            print(f"  [{i+1:02d}/{NUM_PAIRS}] ✘  erro: {exc}")

    random.shuffle(pairs)
    split_idx = int(len(pairs) * TRAIN_RATIO)
    train_pairs = pairs[:split_idx]
    test_pairs  = pairs[split_idx:]

    for filepath, dataset in [(TRAIN_FILE, train_pairs), (TEST_FILE, test_pairs)]:
        with open(filepath, "w", encoding="utf-8") as f:
            for item in dataset:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"[INFO] Salvo: {filepath}  ({len(dataset)} exemplos)")

    print(f"\n[OK] Dataset gerado: {len(train_pairs)} treino | {len(test_pairs)} teste")


if __name__ == "__main__":
    main()