# pepe_checker

A simple deep learning project to detect Pepe images using TensorFlow/Keras.

## Project Structure

- `pepe(2).ipynb`: Main notebook for data loading, preprocessing, model
  training, and evaluation.
- `pepecheck/`: Directory containing image data.

## Setup

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Run the notebook in Jupyter or VS Code.

## Dependencies

See `requirements.txt` for all required packages.

## Usage

- Place your Pepe and non-Pepe images in the `pepecheck/` directory as per the
  notebook instructions.
- Run the notebook to train and evaluate the model.

## Features

- Loads and preprocesses image data
- Builds and trains a CNN for binary classification
- Evaluates model performance
- Visualizes training progress

## Improvements

- Modularize code into functions for reusability
- Add model checkpointing and early stopping
- Improve documentation and error handling
- Add more tests and validation

---
