import torch
import argparse
import pickle

# Merge two pth files
def merge_weights(input_path1, input_path2, output_path):
    """
    Merge two pth files
    
    Args:
        input_path1: Path to input weights file
        input_path2: Path to input weights file
        output_path: Path to save merged weights
    """
    print(f"Loading weights from {input_path1}")
    # Check if it is a pkl file, read with pickle if so
    if input_path1.endswith('.pkl'):
        with open(input_path1, 'rb') as f:
            obj = f.read()
        weights1 = {key: weight_dict for key, weight_dict in pickle.loads(obj, encoding='latin1').items()}    
    else:
        weights1 = torch.load(input_path1, map_location='cpu')
    
    print(f"Loading weights from {input_path2}")
    if input_path2.endswith('.pkl'):
        with open(input_path2, 'rb') as f:
            obj = f.read()
        weights2 = {key: weight_dict for key, weight_dict in pickle.loads(obj, encoding='latin1').items()}    
    else:
        weights2 = torch.load(input_path2, map_location='cpu')
    
    if "model" in weights1:
        weights1 = weights1["model"]
    if "model" in weights2:
        weights2 = weights2["model"]

    # Print the keys of the two weights
    print("weights1 keys: ", weights1.keys())
    print("weights2 keys: ", weights2.keys())
    
    new_state_dict = {}

    # Merge two weight dictionaries
    new_state_dict.update(weights1)
    # If the key of the first weight does not start with 'student', add it
    for key in weights1.keys():
        if not key.startswith("student"):
            new_state_dict["student." + key] = weights1[key]
        else:
            new_state_dict[key] = weights1[key]
    # Add 'teacher' prefix to the second weight if it does not start with 'teacher'
    for key in weights2.keys():
        if not key.startswith("teacher"):
            new_state_dict["teacher." + key] = weights2[key]
        else:
            new_state_dict[key] = weights2[key]
        if "level_embeds" in key: # In KD-DETR, level_embeds also needs to be added to the student model
            new_state_dict["student." + key] = weights2[key]

    print(f"Merged weights count: {len(new_state_dict)}")
    print(f"Saving merged weights to {output_path}")

    print("new keys:", new_state_dict.keys())
    
    torch.save(new_state_dict, output_path)
    print("Merging completed successfully!")

def main():
    parser = argparse.ArgumentParser(description='Convert PyTorch weights by removing glee. prefix')
    parser.add_argument('--input1', type=str, required=True, help='Input weights file path')
    parser.add_argument('--input2', type=str, required=False, help='Input weights file path')
    parser.add_argument('--output', type=str, required=True, help='Output weights file path')
    
    args = parser.parse_args()
    merge_weights(args.input1, args.input2, args.output)

if __name__ == '__main__':
    main()