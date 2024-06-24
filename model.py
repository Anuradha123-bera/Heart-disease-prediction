import os
from joblib import load

# Get the path to the current directory
current_directory = os.path.dirname(os.path.abspath(__file__))

# Specify the filename of the model file
model_filename = 'Random_forest_model.joblib'

# Construct the full path to the model file
model_path = os.path.join(current_directory, model_filename)

# Check if the model file exists
if os.path.exists(model_path):
    # Load the model
    model = load(model_path)
    print("Model loaded successfully!")
else:
    print(f"Model file '{model_filename}' not found in the directory: {current_directory}")
