import json

# Read the original JSON file
input_file = "data/ruozhiba_qaswift.json"  # Your JSON file name
output_file = "data/ruozhiba_qaswift_train.json"  # Output JSON file name

with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# Converted data
converted_data = []

for item in data:
    converted_item = {
        "instruction": item["query"],
        "input": "",
        "output": item["response"]
    }
    converted_data.append(converted_item)

# Save as a JSON file (outermost layer is a list)
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(converted_data, f, ensure_ascii=False, indent=4)

print(f"Conversion completed, data has been saved to {output_file}")  # Conversion completion message