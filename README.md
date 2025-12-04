# MetTrain

This repository contains the implementation of Metamorphic Training for Natural Language Inference (NLI) tasks. This project utilizes Large Language Models (LLMs) to generate Metamorphic samples for augmenting training data and unlabeled data to leverage improving model performance.

> **⚠️ Disclaimer**
>
> The code in this repository is currently a research prototype and may appear disorganized, containing many experimental scripts and temporary files. We plan to refactor and clean up the codebase comprehensively after the paper is accepted. Thank you for your understanding.

## 🛠️ Environment Setup

### 1. Basic Environment
Please ensure that Python 3.8+ is installed.

The project is developed using **PyTorch 1.12.1** with **CUDA 11.3**. While newer versions may work, we recommend using the specified versions for reproducibility.

You can install all dependencies using the provided requirements file:
```bash
pip install -r requirements-backup.txt
```

Alternatively, you can install the key dependencies manually:

```bash
# Install PyTorch (Specific version used in development)
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

# Install other key dependencies
pip install transformers==4.29.2 scikit-learn==1.6.1 numpy==1.26.4 tqdm==4.67.1 pandas==2.2.3 openai==1.75.0 ollama==0.4.8 matplotlib==3.9.4 seaborn==0.13.2
```


### 2. LLM Setup
This project relies on Large Language Models  to generate metamorphic data. The code communicates with locally running LLM services (such as Ollama or LM Studio) via `LMstudio.py` by default.

1.  **Install and run Ollama** (or other local services compatible with the OpenAI API format).
2.  **Pull the model**: e.g., `llama-3-8b`.
3.  **Configure connection**: Check the `HOST`, `PORT`, and `MODEL` variables in `LMstudio.py` to ensure they point to your local service.

```python
# Example configuration in LMstudio.py
HOST = "localhost"
PORT = 1234  # or 11434 for Ollama
MODEL = "llama-3-8b"
```

## 🚀 Quick Start

### Training

The main entry point is `NLImain.py`. You can control the training process via command-line arguments.

#### Basic Training Example
```bash
python NLImain.py --dataset_name snli --model_name bert-base-uncased --epochs 10
```

#### Training with Data Augmentation
```bash
python NLImain.py --dataset_name snli --use_data_augmentation --labeled_ratio 0.1
```

#### Detailed Arguments Explanation

**Dataset & Model Configuration**
*   `--dataset_name`: The name of the NLI dataset to use (e.g., `snli`, `mnli`). Default is `snli`.
*   `--model_name`: The Hugging Face pre-trained model identifier (e.g., `bert-base-uncased`).
*   `--llm_model_name`: The name of the Large Language Model used for generating metamorphic samples.

**Training Hyperparameters**
*   `--epochs`: Total number of training epochs. Default is `25`.
*   `--learning_rate`: The learning rate for optimization. Default is `1e-5`.
*   `--batch_size`: The batch size for training and evaluation. Default is `32`.

**Data Management**
*   `--labeled_ratio`: The proportion of the training set to treat as labeled data (e.g., `0.005` for 0.5%).
*   `--unlabeled_ratio`: The proportion of the training set to treat as unlabeled data.
*   `--val_ratio`: The proportion of data to use for validation if not using the original validation set.
*   `--used_original_val`: Flag to use the official validation set from the dataset. Default is `True`.
*   `--use_data_augmentation`: Flag to enable Metamorphic Data Augmentation during training.

**Training Strategies**
*   `--progressive_training`: Enables progressive training where the model is trained in stages or with curriculum learning.
*   `--unlabeled_first_training`: Enables a strategy where the model is first trained on unlabeled data (e.g., via consistency loss) before labeled data.
*   `--only_unlabeled`: Experimental flag to train using *only* unlabeled data.

**Execution & Hardware**
*   `--only_test`: If set, skips training and only runs evaluation. Requires `--model_load_path`.
*   `--model_save_path`: Custom path to save the trained model.
*   `--model_load_path`: Path to a checkpoint to load for testing or resuming training.
*   `--device`: Compute device to use (`cpu`, `cuda`, `cuda:0`, or `auto`).
*   `--gpu`: Specific GPU ID to use (e.g., `0`, `1`).

### Metamorphic Augmentation

The core data augmentation logic is located in `MRAugment.py`. This module defines various Metamorphic Relations (MRs), such as:
*   **Invariant**: Synonym replacement (`mr_synonym_replacement`), voice switch (`mr_voice_switch`), etc.
*   **Flip**: Negation flip (`mr_negation_flip`), antonym substitution (`mr_antonym_substitution`), etc.
*   **Neutral**: Adding uninformative information (`mr_uninformative`), etc.

These functions generate augmented samples by calling LLMs via Prompt Engineering.

#### ⚠️ Automatic Data Generation
When you run `NLImain.py` with `--use_data_augmentation` for the first time:
1.  The code checks if augmented data exists in `nli_aug_data/`.
2.  If not found, it **automatically triggers the generation process** using the configured LLM.
3.  **This process can take a significant amount of time** depending on your dataset size and LLM inference speed.
4.  **Ensure your LLM service (e.g., Ollama) is running** before starting the training script.
5.  Generated data is saved locally and will be reused in subsequent runs.

### Training Workflow
The default training pipeline (`train_nli` in `NLItrain.py`) typically consists of three stages:
1.  **Labeled Training**: Train the model on the small labeled dataset.
2.  **Dual Consistency Training**: Fine-tune using the augmented data with consistency loss (leveraging Metamorphic Relations).
3.  **Unlabeled Training**: Further training using unlabeled data (if provided).

## 📂 Directory Organization

To ensure the code runs correctly, please organize your files as follows or update the paths in the code accordingly.

### Recommended Structure
```text
MTrain/
├── NLImain.py
├── MRAugment.py
├── ... (other python scripts)
├── nli_dataset_cache/      <-- Cache for HuggingFace datasets
├── nli_aug_data/           <-- Storage for augmented data
└── nli_saved_model/        <-- Default path for saving trained models
```

### ⚠️ Important Configuration Note
Currently, some paths are hardcoded in `NLIdataset.py`. Before running the code, please open `NLIdataset.py` and modify the following variables to match your local environment:

```python
# In NLIdataset.py

# Change this to your desired path for storing augmented data
DEFAULT_AUG_ROOT = "./nli_aug_data" 

# Change this to your desired path for caching datasets
cache_dir = "./nli_dataset_cache"
```

### Model Storage
*   **Pre-trained Models**: By default, the code uses Hugging Face's `AutoModel` which downloads models to the default Hugging Face cache (`~/.cache/huggingface`). You can also specify a local path using `--model_name`.
*   **Trained Checkpoints**: Trained models will be saved to `nli_saved_model/{dataset_name}/{model_name}/` by default, unless `--model_save_path` is specified.

## 📂 Project Structure

*   `NLImain.py`: Main entry point, handling argument parsing and flow control.
*   `MRAugment.py`: **Core file**, implementing various LLM-based metamorphic relation generation logic.
*   `NLItrain.py` / `NLI_train_stage.py`: Contains training loops and specific training strategies (e.g., metamorphic relation loss functions).
*   `NLIdataset.py`: Dataset loading and preprocessing.
*   `NLImodel.py`: Model definition.
*   `LMstudio.py`: Interface responsible for communicating with the LLM backend.
*   `metamorphic_training_usage.py`: Example showing how to integrate metamorphic training into the code.

## 📝 Citation

If this code is helpful for your research, please cite our work after the paper is published. (To be added)
