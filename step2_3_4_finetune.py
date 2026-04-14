"""Passos 2, 3 e 4: Pipeline completo de Fine-Tuning com QLoRA
  - Passo 2: Quantização 4-bit via BitsAndBytes (nf4 + float16)
  - Passo 3: LoRA (r=64, alpha=16, dropout=0.1, tarefa CAUSAL_LM)
  - Passo 4: Treinamento com SFTTrainer + paged_adamw_32bit + cosine LR"""

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, TaskType
from trl import SFTTrainer

# Configurações gerais
BASE_MODEL = "meta-llama/Meta-Llama-3-8B"   # Modelo base (necessita acesso HF)
OUTPUT_DIR      = "./results"
ADAPTER_DIR     = "./lora_adapter"
TRAIN_FILE      = "data/train.jsonl"
TEST_FILE       = "data/test.jsonl"
MAX_SEQ_LENGTH  = 512


# PASSO 2 — Quantização 4-bit (QLoRA)
def get_bnb_config() -> BitsAndBytesConfig:
    """
    Configura o BitsAndBytes para carregar o modelo em 4-bits usando NF4
    (NormalFloat 4-bit). O compute_dtype float16 garante que as operações
    de forward/backward sejam realizadas em meia precisão, economizando
    memória sem grande perda de qualidade numérica.
    """
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",          # NormalFloat 4-bit
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,     # Quantização dupla para ainda menos memória
    )


# PASSO 3 — Arquitetura LoRA
def get_lora_config() -> LoraConfig:
    """
    Configura o LoRA:
      - r (rank) = 64  → dimensão das matrizes de decomposição
      - lora_alpha = 16 → fator de escala dos novos pesos
      - lora_dropout = 0.1 → evita overfitting
      - task_type = CAUSAL_LM → adaptação para geração de texto
    As camadas alvo ('target_modules') são as projeções de atenção
    típicas do Llama-2.
    """
    return LoraConfig(
        r=64,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

# PASSO 4 — Pipeline de Treinamento e Otimização
def get_training_arguments() -> TrainingArguments:
    """
    Argumentos de treinamento com engenharia do otimizador para GPUs limitadas:
      - optimizer: paged_adamw_32bit → transfere picos de memória GPU → CPU
      - lr_scheduler_type: cosine   → decaimento suave da taxa de aprendizado
      - warmup_ratio: 0.03          → aquecimento nos primeiros 3% dos passos
    """
    return TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,         # Batch efetivo = 4 × 4 = 16
        optim="paged_adamw_32bit",             # Otimizador paginado (requisito)
        save_steps=50,
        logging_steps=10,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=True,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,                     # Warmup ratio (requisito)
        group_by_length=True,
        lr_scheduler_type="cosine",            # Cosine scheduler (requisito)
        report_to="none",                      # Troque para "wandb" se quiser logging
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )


def format_instruction(sample: dict) -> str:
    """Formata o par instruction/response no template Alpaca."""
    return (
        "### Instrução:\n"
        f"{sample['instruction']}\n\n"
        "### Resposta:\n"
        f"{sample['response']}"
    )


def main():
    # Carrega datasets 
    print("[INFO] Carregando datasets...")
    dataset = load_dataset(
        "json",
        data_files={"train": TRAIN_FILE, "test": TEST_FILE},
    )

    # Modelo e Tokenizer 
    print("[INFO] Carregando tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("[INFO] Carregando modelo base com quantização 4-bit (QLoRA)...")
    bnb_config = get_bnb_config()
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # LoRA Config 
    print("[INFO] Configurando LoRA (r=64, alpha=16, dropout=0.1)...")
    lora_config = get_lora_config()

    # Training Arguments 
    training_args = get_training_arguments()

    # SFTTrainer 
    print("[INFO] Inicializando SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        peft_config=lora_config,
        formatting_func=format_instruction,
        max_seq_length=MAX_SEQ_LENGTH,
        tokenizer=tokenizer,
        args=training_args,
        packing=False,
    )

    # Treinamento 
    print("[INFO] Iniciando treinamento...")
    trainer.train()

    # Salva o adaptador LoRA 
    print(f"[INFO] Salvando adaptador LoRA em '{ADAPTER_DIR}'...")
    trainer.model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)
    print("[OK] Pipeline concluído com sucesso!")


if __name__ == "__main__":
    main()
