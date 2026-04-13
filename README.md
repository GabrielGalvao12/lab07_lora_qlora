# Laboratório 07 — Especialização de LLMs com LoRA e QLoRA

**iCEV — Instituto de Ensino Superior**  
Disciplina: Tópicos em Inteligência Artificial  
Atividade: Laboratório 07

## Objetivo

Construir um pipeline completo de *fine-tuning* de um modelo de linguagem
fundacional (Llama-2 7B) utilizando técnicas de eficiência de parâmetros
(**PEFT/LoRA**) e quantização (**QLoRA**) para viabilizar o treinamento em
hardwares limitados.

## Estrutura do Repositório

```
lab07_lora_qlora/
│
├── data/                          # Gerado automaticamente pelo Passo 1
│   ├── train.jsonl                # 90% dos pares instruction/response
│   └── test.jsonl                 # 10% dos pares instruction/response
│
├── results/                       # Checkpoints gerados pelo SFTTrainer
├── lora_adapter/                  # Adaptador LoRA salvo após o treinamento
│
├── step1_generate_dataset.py      # Passo 1 — Geração de dataset sintético
├── step2_3_4_finetune.py          # Passos 2, 3 e 4 — Pipeline QLoRA completo
├── requirements.txt               # Dependências do projeto
└── README.md                      # Este arquivo
```

## Configurações Técnicas Implementadas

### Passo 1 — Geração de Dataset
O dataset sintético foi gerado via **API do Google Gemini (gemini-1.5-flash)**,
conforme orientação do professor de que qualquer API generativa é aceita para
esta etapa. Foram gerados 55 pares `instruction`/`response` no domínio de
Ciência de Dados e Machine Learning, divididos em 90% treino e 10% teste.

### Passo 2 — Quantização (QLoRA)
| Parâmetro                  | Valor                     |
|----------------------------|---------------------------|
| Tipo de quantização        | `nf4` (NormalFloat 4-bit) |
| `load_in_4bit`             | `True`                    |
| `compute_dtype`            | `float16`                 |
| Double quantization        | `True`                    |

### Passo 3 — LoRA
| Hiperparâmetro          | Valor                        |
|-------------------------|------------------------------|
| Rank `r`                | **64**                       |
| Alpha `lora_alpha`      | **16**                       |
| Dropout `lora_dropout`  | **0.1**                      |
| Task type               | `CAUSAL_LM`                  |

### Passo 4 — Otimizador e Scheduler
| Parâmetro               | Valor                        |
|-------------------------|------------------------------|
| Otimizador              | **`paged_adamw_32bit`**      |
| LR Scheduler            | **`cosine`**                 |
| Warmup Ratio            | **0.03**                     |
| Learning Rate           |  `2e-4`                      |

## Requisitos de Hardware

- GPU com pelo menos **16 GB VRAM** (recomendado: Google Colab com T4/A100,
  ou Kaggle GPU)
- Acesso ao modelo `meta-llama/Llama-2-7b-hf` no Hugging Face

## Como Executar

### 1. Clonar o repositório e instalar dependências

```bash
git clone <URL_DO_REPOSITORIO>
cd lab07_lora_qlora
pip install -r requirements.txt
```

### 2. Configurar variáveis de ambiente

```bash
export GEMINI_API_KEY="AIza..."    # Chave gratuita em aistudio.google.com
export HF_TOKEN="hf_..."           # Token Hugging Face
huggingface-cli login
```

### 3. Gerar o dataset sintético (Passo 1)

```bash
python step1_generate_dataset.py
```

### 4. Executar o fine-tuning QLoRA (Passos 2, 3 e 4)

```bash
python step2_3_4_finetune.py
```

### Política de Uso de IA

Partes geradas/complementadas com IA, revisadas por Gabriel.

Ferramentas de IA generativa foram utilizadas como IA generativa Claude (Anthropic), suporte na geração de
templates de código e estrutura inicial dos scripts. Todo o conteúdo foi
revisado criticamente e validado antes da submissão, em conformidade com
a Regra de Ouro do contrato pedagógico do iCEV.

## Referências

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — Hu et al., 2021  
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) — Dettmers et al., 2023  
- [Documentação PEFT — Hugging Face](https://huggingface.co/docs/peft)  
- [Documentação TRL — SFTTrainer](https://huggingface.co/docs/trl/sft_trainer)  
- [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes)
- [Google Gemini API](https://aistudio.google.com)