import torch

# Load your PyTorch model
model_path = "YOUR_MODEL_PATH.pt"
torch_model = torch.load(model_path)

# Set the model to evaluation mode
torch_model.eval()

# Provide example input to the model (input_shape should match the input shape of your model)
example_input = torch.randn(49) # observation dimension: agile policy 61 dim, ra 19dim, recovery policy 49dim

# Export the model to ONNX format
path_list = model_path.split("/")
model_dir = ""
for i in range(len(path_list) - 1):
    model_dir += path_list[i]
    model_dir += "/"
torch.onnx.export(torch_model, example_input, '{}{}.onnx'.format(model_dir, model_path.split("/")[-1].split('.')[0]), verbose=True, input_names=['obs'], output_names=['action'])
