import json
import re
import os

class DictionaryHelper:
    def __init__(self, 
                 dict_path="vocab/englishindex_cleaned.json", 
                 stopword_path="vocab/english_stopwords.txt"):
        
        self.glossary = {}
        self.stopwords = set()
        
        self._load_stopwords(stopword_path)
        
        self._load_and_optimize_dictionary(dict_path)

    def _load_stopwords(self, path):
        #loads line-separated stopwords from text file.
        if not os.path.exists(path):
            print(f"Warning: Stopword file not found at {path}. Using empty set.")
            return

        with open(path, "r", encoding="utf-8") as f:
            self.stopwords = set(line.strip().lower() for line in f if line.strip())

    def _load_and_optimize_dictionary(self, path):
        # loads the JSON dictionary and applies 'Winner-Takes-All' logic
        # to keep only the highest frequency translation for non-stopwords.
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dictionary file not found at {path}")

        print(f"Loading dictionary from {path}...")
        with open(path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        iterator = raw_data.items() if isinstance(raw_data, dict) else raw_data

        count = 0
        for eng_term, candidates in iterator:
            eng_term = eng_term.lower().strip()

            if eng_term in self.stopwords:
                continue

            if not candidates:
                continue

            try:
                best_candidate = sorted(candidates, key=lambda x: x[1], reverse=True)[0]
                best_translation = best_candidate[0]
                
                self.glossary[eng_term] = best_translation
                count += 1
            except (IndexError, TypeError):
                continue

        print(f"Dictionary optimized. Loaded {count} terms (filtered stopwords).")

    def get_hints(self, sentence):
        # generates the prompt prefix string for a given source sentence.

        if sentence is None:
            return ""

        hints = []
        tokens = re.findall(r'\b[a-zA-Z]+\b', sentence.lower())
        
        seen_hints = set()

        for token in tokens:
            if token in self.glossary and token not in seen_hints:
                hints.append(f"{token}: {self.glossary[token]}")
                seen_hints.add(token)
        
        if not hints:
            return ""
        
        return "Dictionary: {" + ", ".join(hints[:3]) + "} "
