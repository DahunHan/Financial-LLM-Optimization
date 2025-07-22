###
# create_test_subset.py
import json

original_file = "data/train.json"
subset_file = "data/train_subset.json"
num_items = 10

print(f"Loading original data from {original_file}...")
try:
    with open(original_file, 'r', encoding='utf-8') as f:
        full_data = json.load(f)

    print(f"Creating a small subset of {num_items} items...")
    subset_data = full_data[:num_items]

    with open(subset_file, 'w', encoding='utf-8') as f:
        json.dump(subset_data, f, indent=2, ensure_ascii=False)

    print(f"\n[SUCCESS] Successfully created '{subset_file}' with the first {num_items} items.")
    print("Now, try running the main script with this new subset file.")

except Exception as e:
    print(f"An error occurred: {e}")