import torch

# Load the file
model_path = 'realsense_team/testingRev1/best.pt'
contents = torch.load(model_path, map_location=torch.device('cpu'))

# Check the type of the loaded object
print(type(contents))