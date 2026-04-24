# MetTrain

This repository contains the implementation of Metamorphic Training for Natural Language Inference (NLI) tasks. This project utilizes Large Language Models (LLMs) to generate Metamorphic samples for augmenting training data and unlabeled data to leverage improving model performance.

> **⚠️ Disclaimer**
>
> The code in this repository is currently a research prototype and may appear disorganized, containing many experimental scripts and temporary files, deprecated code, legacy versions, duplicated implementations, and unused experimental scripts. Please treat the codebase as a prototype and prioritize the main pipeline files described below when reproducing results. We plan to refactor and clean up the codebase comprehensively after the paper is accepted. Thank you for your understanding. 

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

This project relies on Large Language Models to generate metamorphic data. The code communicates with locally running LLM services, such as Ollama or LM Studio, via `LMstudio.py` by default.

1. **Install and run Ollama** or other local services compatible with the OpenAI API format.
2. **Pull the model**, e.g., `llama-3-8b`.
3. **Configure connection**: Check the `HOST`, `PORT`, and `MODEL` variables in `LMstudio.py` to ensure they point to your local service.

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
python NLImain.py --dataset_name snli --use_data_augmentation --labeled_ratio 0.0037 --llm_model_name phi-4
```

#### Alternative Entrypoints (Optional)

In addition to `NLImain.py`, you can also run experiments via:

```bash
python new_run.py
python new_test.py
```

These scripts are optional alternatives for some experiment workflows and testing scenarios.

#### Detailed Arguments Explanation

**Dataset & Model Configuration**

* `--dataset_name`: The name of the NLI dataset to use, e.g., `snli`, `mnli`. Default is `snli`.
* `--model_name`: The Hugging Face pre-trained model identifier, e.g., `bert-base-uncased`.
* `--llm_model_name`: The name of the Large Language Model used for generating metamorphic samples.

**Training Hyperparameters**

* `--epochs`: Total number of training epochs. Default is `25`.
* `--learning_rate`: The learning rate for optimization. Default is `1e-5`.
* `--batch_size`: The batch size for training and evaluation. Default is `32`.

**Data Management**

* `--labeled_ratio`: The proportion of the training set to treat as labeled data, e.g., `0.0037` for 0.37%.
* `--unlabeled_ratio`: The proportion of the training set to treat as unlabeled data.
* `--val_ratio`: The proportion of data to use for validation if not using the original validation set.
* `--used_original_val`: Flag to use the official validation set from the dataset. Default is `True`.
* `--use_data_augmentation`: Flag to enable Metamorphic Data Augmentation during training.

**Training Strategies**

* `--progressive_training`: Enables progressive training where the model is trained in stages or with curriculum learning.
* `--unlabeled_first_training`: Enables a strategy where the model is first trained on unlabeled data, e.g., via consistency loss, before labeled data.
* `--only_unlabeled`: Experimental flag to train using *only* unlabeled data.

**Execution & Hardware**

* `--only_test`: If set, skips training and only runs evaluation. Requires `--model_load_path`.
* `--model_save_path`: Custom path to save the trained model.
* `--model_load_path`: Path to a checkpoint to load for testing or resuming training.
* `--device`: Compute device to use, such as `cpu`, `cuda`, `cuda:0`, or `auto`.
* `--gpu`: Specific GPU ID to use, e.g., `0`, `1`.

### Metamorphic Augmentation

The core data augmentation logic is located in `MRAugment.py`. This module defines various Metamorphic Relations (MRs), such as:

* **Invariant**: Synonym replacement (`mr_synonym_replacement`), voice switch (`mr_voice_switch`), etc.
* **Flip**: Negation flip (`mr_negation_flip`), antonym substitution (`mr_antonym_substitution`), etc.
* **Neutral**: Adding uninformative information (`mr_uninformative`), etc.

These functions generate augmented samples by calling LLMs via prompt engineering.

#### ⚠️ Automatic Data Generation

When you run `NLImain.py` with `--use_data_augmentation` for the first time:

1. The code checks if augmented data exists in the configured augmented-data directory.
2. If not found, it **automatically triggers the generation process** using the configured LLM.
3. **This process can take a significant amount of time** depending on your dataset size and LLM inference speed.
4. **Ensure your LLM service, e.g., Ollama, is running** before starting the training script.
5. Generated or prepared data is saved locally and will be reused in subsequent runs. If the required JSON files already exist, such as the provided cropped SNLI example data, the code will load them directly instead of regenerating them.

#### ✅ Using the Provided Cropped SNLI Example Dataset

We have also provided an example dataset for `snli` to help users run the training pipeline without generating data from scratch.

Due to file upload size limitations, the full original SNLI source dataset could not be uploaded to this repository. Instead, we provide a **cropped version** of the SNLI-based prepared data as a runnable example.

The provided cropped SNLI example dataset is located at:

```text
MetTrain/nli_arg_data/snli_lr0.0037_phi-4/
```

If you do not want to use an LLM to generate metamorphic data yourself, you can directly use the provided cropped SNLI example dataset for training. Make sure `NLIdataset.py` points to the correct data root:

```python
DEFAULT_AUG_ROOT = "./nli_arg_data"
```

Then run training with arguments matching the example-data directory name:

```bash
python NLImain.py --dataset_name snli --use_data_augmentation --labeled_ratio 0.0037 --llm_model_name phi-4
```

When the required JSON files already exist under `MetTrain/nli_arg_data/snli_lr0.0037_phi-4/`, the code will load the prepared data directly instead of triggering LLM-based generation.

## 📂 Directory Organization

To ensure the code runs correctly, please organize your files as follows or update the paths in the code accordingly.

### Recommended Structure

```text
MetTrain/
├── NLImain.py
├── MRAugment.py
├── ... (other python scripts)
├── nli_dataset_cache/      <-- Cache for Hugging Face datasets
├── nli_arg_data/           <-- Provided cropped SNLI example data
│   └── snli_lr0.0037_phi-4/
│       ├── labeled_data.json
│       ├── unlabeled_data.json
│       ├── augmented_data.json
│       ├── test.json
│       └── validation.json
└── nli_saved_model/        <-- Default path for saving trained models
```

### ⚠️ Important Configuration Note

Currently, some paths are hardcoded in `NLIdataset.py`. Before running the code, please open `NLIdataset.py` and modify the following variables to match your local environment:

```python
# In NLIdataset.py

# Change this to your desired path for storing augmented/prepared data.
# For the provided cropped SNLI example dataset, use:
DEFAULT_AUG_ROOT = "./nli_arg_data"

# Change this to your desired path for caching datasets
cache_dir = "./nli_dataset_cache"
```

### Dataset JSON Format and Location

If you use local prepared data, dataset files should be stored in **JSON format** under the augmented-data directory expected by `NLIdataset.py`.

For the provided cropped SNLI example dataset, the root directory should be:

```text
MetTrain/nli_arg_data/
```

Expected subdirectory naming rule:

```text
{dataset_name}_lr{labeled_ratio}_{llm_model_name}
```

The provided cropped SNLI example follows this structure:

```text
MetTrain/nli_arg_data/snli_lr0.0037_phi-4/
```

Expected JSON files in that directory include:

* `labeled_data.json`
* `unlabeled_data.json`
* `augmented_data.json`
* `test.json`
* `validation.json`

#### Provided SNLI Example Data

This repository includes a cropped SNLI example dataset following the same JSON format above.

The original SNLI source dataset is not uploaded because it exceeds the file upload size limit. Therefore, we provide a cropped version that can be used as an example for quickly running and testing the training pipeline.

If you do not want to generate metamorphic data with an LLM, you can use this provided cropped SNLI data directly. Please ensure that:

1. The JSON files are placed under `MetTrain/nli_arg_data/snli_lr0.0037_phi-4/`.
2. `DEFAULT_AUG_ROOT` in `NLIdataset.py` is set to `./nli_arg_data`.
3. The command-line arguments match the directory name:

```bash
python NLImain.py --dataset_name snli --use_data_augmentation --labeled_ratio 0.0037 --llm_model_name phi-4
```

If your data is not placed in this structure, update `DEFAULT_AUG_ROOT` and related path settings in `NLIdataset.py` before running training.

### Model Storage

* **Pre-trained Models**: By default, the code uses Hugging Face's `AutoModel` which downloads models to the default Hugging Face cache, `~/.cache/huggingface`. You can also specify a local path using `--model_name`.
* **Trained Checkpoints**: Trained models will be saved to `nli_saved_model/{dataset_name}/{model_name}/` by default, unless `--model_save_path` is specified.

## 📂 Project Structure

* `NLImain.py`: Main entry point, handling argument parsing and flow control.
* `MRAugment.py`: **Core file**, implementing various LLM-based metamorphic relation generation logic.
* `NLItrain.py` / `NLI_train_stage.py`: Contains training loops and specific training strategies, e.g., metamorphic relation loss functions.
* `NLIdataset.py`: Dataset loading and preprocessing.
* `NLImodel.py`: Model definition.
* `LMstudio.py`: Interface responsible for communicating with the LLM backend.
* `metamorphic_training_usage.py`: Example showing how to integrate metamorphic training into the code.

## 📝 Citation

If this code is helpful for your research, please cite our work after the paper is published. (To be added)
