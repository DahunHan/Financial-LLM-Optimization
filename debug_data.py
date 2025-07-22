###
# debug_data.py
import json
from tqdm import tqdm

local_data_path = "data/train.json"
print(f"Starting to debug the data structure of {local_data_path}...")

malformed_items_found = 0

try:
    with open(local_data_path, 'r', encoding='utf-8') as f:
        finqa_data = json.load(f)

    print(f"Successfully loaded the JSON file. It contains {len(finqa_data)} items.")
    print("Now, checking the type of each item in the list...")

    # Use tqdm for a progress bar
    for i, item in enumerate(tqdm(finqa_data, desc="Checking data types")):
        # Check if the item in the list is NOT a dictionary
        if not isinstance(item, dict):
            malformed_items_found += 1
            # If it's not a dictionary, print its details
            print(f"\n--- Found malformed data at index {i} ---")
            print(f"Type: {type(item)}")
            print(f"Content: {item}")
            print("----------------------------------------")

    if malformed_items_found == 0:
        print("\n[SUCCESS] All 6,251 items in the list are correctly structured as dictionaries.")
    else:
        print(f"\n[ERROR] Found {malformed_items_found} total malformed items.")


except Exception as e:
    print(f"An error occurred during the debug script execution: {e}")