import os
import json

def remove_segmentation(data):
    """Remove the 'segmentation' key from each dictionary in the list."""
    for item in data:
        if 'segmentation' in item:
            del item['segmentation']
    return data

# Specify the directory containing the JSON files
json_directory = "log/SAMg_sg_06_promptmode-self_grounding_scoremode-normal_weightscores-true"

# Iterate over all files in the directory
for filename in os.listdir(json_directory):
    if filename.endswith(".json"):
        # Full path to the current JSON file
        json_path = os.path.join(json_directory, filename)
        
        # Open and read the JSON file
        with open(json_path, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
        
        # Remove the 'segmentation' key
        modified_data = remove_segmentation(data)
        
        # Define the output path with '_detection' added to the filename
        output_filename = filename.replace('.json', '_detection.json')
        output_path = os.path.join(json_directory, output_filename)
        
        # Save the modified JSON content
        with open(output_path, 'w', encoding='utf-8') as output_file:
            json.dump(modified_data, output_file, indent=4)
        
        print(f"Processed {filename} and saved as {output_filename}")
