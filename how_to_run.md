# How to Run INLP Assignment 3 on Google Colab

This guide explains how to set up and run this project in Google Colab using the provided `ass.zip` file.

## 1. Setup Environment

1. Download the `ass.zip` file containing the code.
2. Upload `ass.zip` to your Google Colab instance (this usually places it in `/content`).
3. Run the following commands in a Colab cell to unzip the code and install the required dependencies:

```bash
# Unzip the code into a folder named "ass" and navigate into it
!unzip -o ass.zip -d /content/ass
%cd /content/ass

# Install the required packages
!pip install -r requirements.txt
```

## 2. API Keys

You have two options for providing your Weights & Biases (wandb) and HuggingFace API keys:

**Option A: Using a `.env` file (Recommended)**
1. Create a file named `.env` locally with the following format:
   ```
   HF_TOKEN=your_huggingface_token
   WANDB_API_KEY=your_wandb_api_key
   ```
2. Upload this `.env` file to the `/content/ass` directory in Colab. The code will automatically load these variables.

**Option B: Setting Environment Variables in Colab**
Run this in a Colab cell before executing your code:
```python
import os
os.environ["HF_TOKEN"] = "your_huggingface_token"
os.environ["WANDB_API_KEY"] = "your_wandb_api_key"
```

## 3. Running the Models

The project uses `main.py` as the entry point. You select a task and a mode (`train`, `evaluate`, or `both`).

Here are the commands you can run in your Colab cells (remember to prepend with `!`):

### Task 1: Cipher Decryption (RNN / LSTM)
```bash
# Train the RNN model
!python main.py task1_rnn --mode train

# Evaluate the RNN model
!python main.py task1_rnn --mode evaluate

# Train the LSTM model
!python main.py task1_lstm --mode train

# Evaluate the LSTM model
!python main.py task1_lstm --mode evaluate
```

### Task 2: Language Modeling (BiLSTM / SSM)
```bash
# Train the BiLSTM model (Masked Language Modeling)
!python main.py task2_bilstm --mode train

# Train the SSM model (Next Word Prediction)
!python main.py task2_ssm --mode train
```

### Task 3: Error Correction Pipeline
```bash
# Run Task 3 with BiLSTM
!python main.py task3_bilstm --mode evaluate

# Run Task 3 with SSM
!python main.py task3_ssm --mode evaluate
```

## 4. Viewing Results

- **Console:** You can see training progress per batch/epoch and the test/validation metrics printed right in Colab output cells.
- **WandB Dashboard:** Follow the link that gets printed when you start a training run to view the interactive charts on Weights & Biases.
- **Output Files:** All model checkpoints (`*_best.pt`), metadata (`*_meta.json`), and final prediction outputs (`*.txt`) are saved in the `outputs/logs/` and `outputs/results/` directories.
