
import torch


def contrastive_search_decode(
    model,
    tokenizer,
    prompt,
    max_new_tokens: int = 50,
    penalty_alpha: float = 0.3,
    top_k: int = 5,
):

    # contrastive search using Hugging Face's remote custom_generate plugin.

    model.eval()
    device = next(model.parameters()).device

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            custom_generate="transformers-community/contrastive-search",
            penalty_alpha=penalty_alpha,
            top_k=top_k,
            trust_remote_code=True,
        )

    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)
