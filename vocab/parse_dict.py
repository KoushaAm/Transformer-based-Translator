import json
import sys

def clean_entry(entry):

    # remove pronunciation from entry.
    # entry format: [word:pronunciation, frequency] -> [word, frequency]

    if not isinstance(entry, list) or len(entry) != 2:
        return entry
    
    word_with_pronunciation = entry[0]
    frequency = entry[1]
    
    if ':' in word_with_pronunciation:
        word = word_with_pronunciation.split(':', 1)[0]
    else:
        word = word_with_pronunciation
    
    return [word, frequency]

def clean_json_file(input_file, output_file):
    # cleaning all entries and save to output file.
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # checiking if it's a dict or list and handle accordingly
    if isinstance(data, dict):
        cleaned_data = {}
        for key, value in data.items():
            if isinstance(value, list):
                cleaned_data[key.strip("!")] = [clean_entry(entry) for entry in value]
            else:
                cleaned_data[key] = value
    elif isinstance(data, list):
        cleaned_data = [clean_entry(entry) for entry in data]
    else:
        return
    
    # cleaned data
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=2)
    
    print(f"Cleaned JSON saved to: {output_file}")
    
    if isinstance(cleaned_data, dict):
        for key in list(cleaned_data.keys())[:3]:
            print(f"\n{key}: (showing first 5 entries)")
            for entry in cleaned_data[key][:5]:
                print(f"  {entry}")
    elif isinstance(cleaned_data, list):
        print("\nFirst 10 cleaned entries:")
        for entry in cleaned_data[:10]:
            print(f"  {entry}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else "englishindex_cleaned.json"
    else:
        import os
        uploads = "/mnt/user-data/uploads"
        json_files = [f for f in os.listdir(uploads) if f.endswith('.json')]
        
        if not json_files:
            sys.exit(1)
        
        input_file = os.path.join(uploads, json_files[0])
        output_file = "/mnt/user-data/outputs/cleaned.json"
    
    print(f"Input: {input_file}")
    print(f"Output: {output_file}\n")
    clean_json_file(input_file, output_file)
