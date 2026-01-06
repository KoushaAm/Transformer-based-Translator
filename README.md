# **Efficient English–Cantonese Low-Resource Machine Translation via LoRA + Contrastive Decoding**

---
## Team
- Jason (Jie Hua Yi)
- Kousha Amouzesh 
- Dixon Snider

## **1. Motivation**

Neural Machine Translation (NMT) has achieved remarkable success for high-resource languages, but it remains brittle in low-resource scenarios where parallel data is scarce. Cantonese represents a unique case of this disparity: despite having over 80 million speakers, authentic written resources are limited due to diglossia, a phenomenon where the spoken language differs significantly from the formal written standard (Standard Written Chinese/Mandarin). Consequently, NMT models frequently fail to capture colloquial Cantonese syntax and vocabulary. This project focuses on improving both the quality and computational efficiency of English-to-Cantonese translation to address these structural deficits.

To achieve this, we leverage mT5-small optimized via Low-Rank Adaptation (LoRA). We explicitly prioritize a lightweight architecture over massive foundation models to address resource constraints and deployment accessibility. While this study focuses on efficiency, we believe that the methods validated in this constrained setting can scale to larger models for further performance gains.

Our early experiments revealed that small models like t5-small can handle high-resource languages like French relatively well (**BLEU ≈ 20**) without tuning. However, when we applied the similar mT5-small model to a low-resource setting, using the `lordjia/Cantonese_English_Translation` dataset, performance collapsed. In the experiments, the base mT5-small model produced almost only the placeholder token `<extra_id_0>` for Cantonese, leading to **BLEU ≈ 0**. 

This showed two things:
1. mT5-small has read Chinese (but predominantly Mandarin) characters in pretraining, but is **not proficient at producing** good Cantonese translations out-of-the-box.
2. There is clear room for improvement via **targeted fine-tuning** and **better decoding**.

We therefore focus on two aspects:
1. **Efficient training** with Low-Rank Adaptation (LoRA), so we can adapt mT5-small to English→Cantonese without full-model fine-tuning.
2. **Higher-quality decoding** with contrastive decoding, in order to avoid degenerate outputs like repeated placeholder tokens and to improve fluency.

This combination lets us study modern techniques (LoRA + contrastive decoding) in a realistic, challenging translation setting.


## **2. Related Work**

### 2.1 Low-Resource Neural Machine Translation for Cantonese

Neural Machine Translation (NMT) underperforms for low-resource languages like Cantonese due to its underrepresented in large multilingual corpora. Liu established the first benchmark for this issue, demonstrating that standard NMT systems struggle with Cantonese-specific syntax and vocabulary without explicit interventions like parallel sentence mining or data augmentation [1]. Building on this, Hong et al. introduced CantonMT, showing that fine-tuning multilingual models on synthetic back-translated data is effective for Cantonese-English translation, with their best NLLB-based model achieving a SacreBLEU score of 16.81, compared to 19.16 for a fine-tuned GPT-4 [2].


### 2.2 Enforcing Lexical Consistency via Data Augmentation

To address the specific issue of "hallucinating" rare Cantonese terms, we draw on methods that enforce lexical constraints through data augmentation. Recent work by Pei et al. on low-resource Manchu translation demonstrated that providing external resources, specifically dictionaries and parallel examples, directly in the context significantly improves translation accuracy where large-scale training data is absent [3]. Unlike traditional constrained decoding algorithms that slow down inference, they found that augmenting the input allows models to "learn" the language dynamically from the provided cues. Inspired by this, we adapt the method by "baking" dictionary hints directly into our training data. This teaches the model to utilize external context as a soft constraint, effectively grounding the translation of rare Cantonese terms without requiring complex modifications to the decoding search process.

### 2.3 Contrastive Decoding for Hallucination Mitigation

Finally, we address the tendency of small models to generate generic or repetitive outputs through Contrastive Decoding (CD). Waldendorf et al. showed that maximizing the log-likelihood difference between a "strong" expert model and a "weak" amateur model significantly reduces hallucinations in large multilingual systems [4]. We implement a variation of this by contrasting our "hint-aware" model (expert) against the "no-hint" baseline (amateur) to amplify the signal of the correct Cantonese vocabulary.

## **3. Approach**

## Model Architecture
We use the **mT5-small** encoder–decoder Transformer model. Translation is performed by prompting the model in the standard format:



The SentencePiece tokenizer from `"google/mt5-small"` is used.

---

## LoRA Fine-Tuning

Instead of updating all parameters, we apply **Low-Rank Adaptation (LoRA)** to the attention modules of the Transformer. Let $W \in \mathbb{R}^{d \times d}$ be a frozen weight matrix. LoRA introduces a trainable low-rank update:

$$
W_{\text{new}} = W + BA
$$

where:
- $A \in \mathbb{R}^{r \times d}$,
- $B \in \mathbb{R}^{d \times r}$,
- and rank $r \ll d$.

Only $A$ and $B$ are trained.

### LoRA configuration (from our training code)
- Rank $r = 64$
- Alpha = 128
- LoRA dropout = 0.05
- Applied to modules: `q`,`v`
- Trained for 5 epochs using HuggingFace Trainer

This approach drastically reduces the number of trainable parameters while still meaningfully adapting the model to the English–Cantonese translation domain.

---

## Decoding Methods Compared

We evaluate four decoding algorithms:

1. **Greedy decoding**  
   Picks the highest-probability token at each step.

2. **Beam search**  
   Explores multiple possible sequences to improve fluency.

3. **Dual-model contrastive decoding (GREEDY_CONTRASTIVE)**
   This is our main decoding method. It uses:
   - a **strong model**: the LoRA-fine-tuned mT5 (“hint” model), and
   - a **weak model**: the base mT5-small.

   The strong model proposes candidate tokens, and the weak model plus a repetition penalty are used to down-weight:
   - tokens that the weak model also finds very likely (generic tokens), and
   - tokens that would lead to repetitive continuations
   

## **4. Data & Preprocessing**

Our experiments use the **lordjia/Cantonese_English_Translation** corpus from Hugging Face, which provides sentence-aligned **English–Cantonese** pairs.

### 4.1 Splits & JSON Format

We load the dataset and split it into **80% train / 10% validation / 10% test**, then save each split as JSON:

- `data/en_yue_train.json`
- `data/en_yue_val.json`
- `data/en_yue_test.json`

Each entry has:

```json
{
  "src": "<english sentence>",
  "tgt": "<cantonese sentence>"
}
```

### 4.2 TranslationDataset Class

We wrap each JSON split using our custom `TranslationDataset class`, which performs:

- Loading the JSON data into memory.

- Building an instruction-style prompt:

    - "translate English to Cantonese"

### 4.3 Dictionary Hints (Optional Prompt Augmentation)

We generate small bilingual glossaries for some training examples using a DictionaryHelper:

- Loads a cleaned English→Cantonese wordlist and removes English stopwords.

- For each source sentence, extracts up to three high-frequency translations of English content words.

- Formats hints like:

```json
Dictionary: {apple: 蘋果, drink: 飲} translate English to Cantonese: <src>
```

- For hint inject, we include a dropout probability of 0.3, where we do not apply the hint. We perform this to ensure that the model does not become over-reliant on the hint structure for generating sentences.

## **5. Code**
Our project implementation consists primarily of our own code, with selective use of open-source libraries and course-provided utilities.

## How to run the code:
### 1. Install Dependencies

Make sure you have Python 3.9+ and install the required libraries:

```bash
pip install -r requirements.txt
```
### 2. (Optional) Prepare the Dataset
This downloads lordjia/Cantonese_English_Translation from Hugging Face and creates
data/en_yue_{train,val,test}.json:
```
python main.py -p
```

### 3. Train and Run a Single Experiment
Each scenario selects a different decoding strategy:

1 → BEAM_HINT (LoRA + beam search)
2 → BEAM_BASE (base mT5 + beam)
3 → GREEDY_CONTRASTIVE (LoRA strong + base weak, default)
4 → GREEDY_HINT (LoRA + greedy)
5 → GREEDY_BASE (base mT5 + greedy)

Example: Train the LoRA “hint” model and run beam search:
```
python main.py --scenario 1
```

If the LoRA model is already trained and you only want to decode:
```
python main.py --scenario 1 -i
```


For a quick test on only 100 examples:

```
python main.py --scenario 3 -s
```
### 4. Evaluate Saved Outputs

To evaluate a single inference file:
```
python main.py --evaluate-only output/inference_results_beam_hint.json
```


To evaluate all JSON outputs in the output/ directory:
```
python main.py -e
```

## Code Written by Our Group
- `main.py` — command-line entry point; sets scenarios, prepares datasets, loads models (base vs LoRA “hint” model), and runs all decoding modes (greedy, beam, contrastive).
- `train.py` — LoRA fine-tuning of mT5-small using HuggingFace’s `Trainer` (including training arguments and checkpoint handling).
- `contrastive.py` — custom **dual-model contrastive decoding** (strong = LoRA model, weak = base model), including token scoring and selection logic.
- `evaluate.py` — utilities for computing BLEU (via sacreBLEU) and extracting model translations from raw outputs.
- `data.py` — `TranslationDataset` implementation: loads JSON splits, builds prompts (`translate English to Cantonese: ...`), injects optional dictionary hints, tokenizes inputs/targets, and masks padding in labels.
- `dictionary.py` — `DictionaryHelper` for loading a cleaned English→Cantonese lexicon, filtering stopwords, and generating per-sentence “Dictionary: {…}” hints.
- `infer-all.sh` — shell script to run all decoding scenarios (greedy, beam, contrastive) over the test set and save outputs.
- Small helper code (in `main.py`) for splitting the Hugging Face dataset and exporting `en_yue_{train,val,test}.json`.

## External Code Used
- `sacrebleu.metrics.BLEU` (provided in CMPT 413 homework starter code)  
- HuggingFace Transformers (model loading, generation, Trainer, tokenizer, etc.)  
- PEFT library for LoRA (`LoraConfig`, `get_peft_model`)  
- HuggingFace `DataCollatorForSeq2Seq`

All contrastive decoding logic and LoRA integration beyond library calls were implemented by our group.



## **References**

[1]	K. Y. Hong, L. Han, R. Batista-Navarro, and G. Nenadic, ‘CantonMT: Cantonese-English Neural Machine Translation Looking into Evaluations’, in Proceedings of the 16th Conference of the Association for Machine Translation in the Americas (Volume 2: Presentations), 2024, pp. 133–144.

[2]	E. K.-Y. Liu, ‘Low-Resource Neural Machine Translation: A Case Study of Cantonese’, in Proceedings of the Ninth Workshop on NLP for Similar Languages, Varieties and Dialects, 2022, pp. 28–40.

[3] R. Pei, Y. Liu, P. Lin, F. Yvon, and H. Schütze, “Understanding In-Context Machine Translation for Low-Resource Languages: A Case Study on Manchu,” arXiv (Cornell University), Feb. 2025, doi: https://doi.org/10.48550/arxiv.2502.11862.

[4] J. Waldendorf, B. Haddow, and A. Birch, “Contrastive Decoding Reduces Hallucinations in Large Multilingual Machine Translation Models,” Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 2526–2539, 2024, doi: https://doi.org/10.18653/v1/2024.eacl-long.155.
