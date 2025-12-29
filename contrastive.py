import torch

def contrastive_decode(
    strong_model,
    weak_model,
    tokenizer,
    strong_prompt,
    weak_prompt=None,
    max_new_tokens=50,
    alpha=0.3,
    top_k=5,
    repetition_penalty=1.0,
    do_sample=False,
    temperature=1.0,
):
    """
    Contrastive Decoding with added Repetition Penalty.
    score = strong_logits - alpha * weak_logits
    """

    strong_model.eval()
    weak_model.eval()

    device = next(strong_model.parameters()).device

    if weak_prompt is None:
        weak_prompt = strong_prompt

    strong_inputs = tokenizer(strong_prompt, return_tensors="pt").to(device)
    weak_inputs = tokenizer(weak_prompt, return_tensors="pt").to(device)

    strong_encoder_out = strong_model.get_encoder()(**strong_inputs)
    weak_encoder_out = weak_model.get_encoder()(**weak_inputs)

    decoder_input_ids = torch.tensor([[tokenizer.pad_token_id]]).to(device)

    for _ in range(max_new_tokens):
        # next token predictions
        strong_logits = strong_model(
            encoder_outputs=strong_encoder_out, decoder_input_ids=decoder_input_ids
        ).logits[:, -1, :]

        weak_logits = weak_model(
            encoder_outputs=weak_encoder_out, decoder_input_ids=decoder_input_ids
        ).logits[:, -1, :]

        strong_logprobs = strong_logits.log_softmax(dim=-1)
        weak_logprobs = weak_logits.log_softmax(dim=-1)

        score = strong_logprobs - alpha * weak_logprobs # contrastive

        if repetition_penalty != 1.0:
            for i in range(decoder_input_ids.size(0)): # batch loop
                previous_tokens = set(decoder_input_ids[i].tolist())
                for token_id in previous_tokens:
                    if score[i, token_id] < 0:
                        score[i, token_id] *= repetition_penalty
                    else:
                        score[i, token_id] /= repetition_penalty

        top_k_scores, top_k_indices = score.topk(top_k, dim=-1)

        if do_sample and top_k > 1:
            probs = torch.softmax(top_k_scores / temperature, dim=-1)
            next_token = top_k_indices.gather(-1, torch.multinomial(probs, 1))
        else:
            next_token = top_k_indices[:, :1]

        if (next_token == tokenizer.eos_token_id).any():
            break

        decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=1)
    return tokenizer.decode(decoder_input_ids[0, 1:], skip_special_tokens=True)
