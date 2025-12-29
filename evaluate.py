import sacrebleu

# source : https://github.com/anoopsarkar/nlp-class-hw/tree/main
bleu = sacrebleu.metrics.BLEU(effective_order=True, tokenize="zh")


def compute_bleu(references, output_data):
    # compute BLEU score between references and predictions.
    if len(references) == 0 or len(output_data) == 0:
        return 0.0

    if len(references) != len(output_data):
        print(
            f"Warning: references ({len(references)}) and outputs ({len(output_data)}) length mismatch"
        )

    score = 0.0
    total = 0.0

    for idx, hypothesis in output_data:
        if idx not in references:
            print(f"Warning: index {idx} not found in references, skipping")
            continue

        ref = references[idx]

        # clean up hypothesis (strip whitespace)
        hypothesis = hypothesis.strip()

        # sacrebleu expects hypothesis as string and references as list of strings
        sentence_score = bleu.sentence_score(hypothesis, ref).score
        score += sentence_score
        total += 1.0

    if total == 0:
        return 0.0

    bleu_score = score / total
    return bleu_score


def evaluate(references, output_data):
    bleu_score = compute_bleu(references, output_data)
    return bleu_score


def extract_translation(generated_text, target_lang="French"):
    # extracting just the translation from the full generated text.

    # Look for the target language marker
    marker = f"{target_lang}:"

    if marker in generated_text:
        # get everything after the last occurrence of the marker
        translation = generated_text.split(marker)[-1].strip()
        # removing any trailing special tokens
        # stopping at newline if present (might indicate next example)
        if "\n" in translation:
            translation = translation.split("\n")[0].strip()
        return translation

    # if marker not found, return the original text stripped
    return generated_text.strip()
