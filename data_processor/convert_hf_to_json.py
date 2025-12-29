import json
from datasets import load_dataset
from tqdm import tqdm

import json
from datasets import load_dataset
from tqdm import tqdm
import random

def convert_to_json(hf_split, lang_pair, out_path):
    src_lang, tgt_lang = lang_pair
    data_list = []

    for ex in hf_split:
        src_txt = ex["translation"][src_lang]
        tgt_txt = ex["translation"][tgt_lang]
        data_list.append({"src": src_txt, "tgt": tgt_txt})

    with open(out_path, "w") as f:
        json.dump(data_list, f, indent=2)
