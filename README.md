# MetTrain

This repository contains the implementation of Metamorphic Training for Natural Language Inference (NLI) tasks. This project utilizes Large Language Models (LLMs) to generate Metamorphic samples for augmenting training data and unlabeled data to leverage improving model performance.

> **⚠️ Disclaimer**
>
> The code in this repository is currently a research prototype and may appear disorganized, containing many experimental scripts and temporary files, deprecated code, legacy versions, duplicated implementations, and unused experimental scripts. Please treat the codebase as a prototype and prioritize the main pipeline files described below when reproducing results. We plan to refactor and clean up the codebase comprehensively after the paper is accepted. Thank you for your understanding.

> **⚠️ Path Configuration Notice**
>
> Some paths in this project may be hardcoded in the scripts, especially in `new_run.py`, `NLIdataset.py`, and related dataset-loading code. Before running any experiment, please carefully check that all dataset paths, cache paths, model paths, and output paths match your local environment.

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
2. **Pull the model**, e.g., `gemma-e-4b-it-qat`.
3. **Configure connection**: Check the `HOST`, `PORT`, and `MODEL` variables in `LMstudio.py` to ensure they point to your local service.

```python
# Example configuration in LMstudio.py
HOST = "localhost"
PORT = 1234  # or 11434 for Ollama
MODEL = "gemma-e-4b-it-qat"
```

> If you only want to train with the provided  RTE example dataset, you do **not** need to generate new data with an LLM.

## 🚀 Quick Start

### Recommended Entry Point: `new_run.py`

For most users, especially when using the provided  RTE example dataset, we recommend using:

```bash
python new_run.py
```

`new_run.py` is the recommended experiment runner because it directly loads prepared JSON or JSONL data files, runs the training pipeline with a fixed configuration dictionary, saves experiment outputs, and avoids accidentally triggering time-consuming LLM-based data generation.

Before running it, open `new_run.py` and modify the `config` section near the bottom of the file.

> **Important:** Some paths in `new_run.py` may still be absolute or hardcoded. Please confirm that `train_data_path`, `val_data_path`, `augmented_dir`, `save_model_dir`, and any dataset list paths are correct before running the code.

For the provided RTE example dataset, use a configuration similar to the following:

```python
config = {
    # Use prepared JSON files directly.
    # This is recommended for the provided  RTE example dataset.
    'use_augmented_data': False,

    # Path to the prepared training file.
    'train_data_path': '/nli_arg_data/rte_lr0.8_gemma-e-4b-it-qat/augmented_data.json',

    # Path to the validation file.
    'val_data_path': '/nli_arg_data/rte_lr0.8_gemma-e-4b-it-qat/validation.json',

    # Dataset information
    'dataset_name': 'rte',
    'labeled_ratio': 0.8,
    'unlabeled_ratio': 0.2,
    'augmented_dir': '/nli_arg_data/rte_lr0.8_gemma-e-4b-it-qat/',
    'llm_model_name': 'gemma-e-4b-it-qat',

    # Model configuration
    'model_name': 'bert-base-uncased',
    'max_length': 128,
    'dropout_rate': 0.3,

    # Training configuration
    'batch_size': 16,
    'learning_rate': 2e-5,
    'weight_decay': 1e-2,
    'warmup_ratio': 0.06,

    # Device configuration
    # 0: cuda:0
    # 1: cuda:1
    # -1: automatically use cuda if available, otherwise cpu
    # You can also use strings such as 'cpu', 'cuda', or 'cuda:0'
    'device': -1,

    # Model saving configuration
    'save_best_model': True,
    'save_model_dir': None,

    # Reproducibility
    'random_seed': 42,
    'shuffle': False,
    'deterministic': True,

    # Experiment epochs
    'baseline_epochs': 0,
    'mr_epochs': 0,
    'full_epochs': 20,
    'ablation_epochs': 0,
}
```

Then make sure the dataset lists in `new_run.py` also point to the provided example data:

```python
train_datasets = [
    '/nli_arg_data/rte_lr0.8_gemma-e-4b-it-qat/augmented_data.json'
]

test_dataset = [
    '/nli_arg_data/rte_lr0.8_gemma-e-4b-it-qat/validation.json'
]
```

If you prefer to use the default augmented file instead, you can also set:

```python
train_datasets = [
    '/nli_arg_data/rte_lr0.8_gemma-e-4b-it-qat/augmented_data.json'
]
```

After modifying the paths, run:

```bash
python new_run.py
```

### Difference Between `new_run.py` and `NLImain.py`

Both `new_run.py` and `NLImain.py` can be used for NLI training, but they serve different purposes.

#### `new_run.py`

`new_run.py` is a configuration-based experiment runner. It is recommended when you already have prepared JSON or JSONL data files.

Main characteristics:

* Directly loads prepared data from `train_data_path` and `val_data_path`.
* Supports JSON and JSONL input files.
* Does not require command-line arguments.
* Does not need to generate new data with an LLM when `use_augmented_data=False`.
* Creates an experiment directory under `experiments/`.
* Saves experiment results, summaries, data analysis files, and training curves.
* Supports random seed control for reproducibility.
* Supports warmup and linear learning-rate scheduling.
* Supports enhanced training strategies such as R-Drop-based training on augmented data.
* Provides more explicit control over model saving, device selection, and experiment configuration.

For the provided  RTE example dataset, `new_run.py` is the preferred way to run training because the data has already been prepared.

#### `NLImain.py`

`NLImain.py` is the original command-line entry point. It is useful when you want to run the default pipeline through command-line arguments.

Example:

```bash
python NLImain.py --dataset_name rte --use_data_augmentation --labeled_ratio 0.8 --llm_model_name gemma-e-4b-it-qat
```

However, when `--use_data_augmentation` is enabled, `NLImain.py` may check for augmented data and trigger LLM-based data generation if the expected files are not found. This can take a long time and requires the local LLM service to be running.

Therefore, for the provided  RTE example dataset, we recommend using `new_run.py` with explicit file paths instead of relying on automatic data detection.

## Training with `NLImain.py`

The main command-line entry point is `NLImain.py`. You can control the training process via command-line arguments.

### Basic Training Example

```bash
python NLImain.py --dataset_name rte --model_name bert-base-uncased --epochs 10
```

### Training with Data Augmentation

```bash
python NLImain.py --dataset_name rte --use_data_augmentation --labeled_ratio 0.8 --llm_model_name gemma-e-4b-it-qat
```

### Alternative Entrypoints

In addition to `NLImain.py`, you can also run experiments via:

```bash
python new_run.py
python new_test.py
```

Among these, `new_run.py` is recommended for prepared example datasets.

## Metamorphic Augmentation

The core data augmentation logic is located in `MRAugment.py`. This module defines various Metamorphic Relations (MRs), such as:

* **Invariant**: Synonym replacement (`mr_synonym_replacement`), voice switch (`mr_voice_switch`), etc.
* **Flip**: Negation flip (`mr_negation_flip`), antonym substitution (`mr_antonym_substitution`), etc.
* **Neutral**: Adding uninformative information (`mr_uninformative`), etc.

These functions generate augmented samples by calling LLMs via prompt engineering.

### ⚠️ Automatic Data Generation

When you run `NLImain.py` with `--use_data_augmentation` for the first time:

1. The code checks if augmented data exists in the configured augmented-data directory.
2. If not found, it **automatically triggers the generation process** using the configured LLM.
3. **This process can take a significant amount of time** depending on your dataset size and LLM inference speed.
4. **Ensure your LLM service, e.g., Ollama, is running** before starting the training script.
5. Generated or prepared data is saved locally and will be reused in subsequent runs. If the required JSON files already exist, such as the provided  RTE example data, the code will load them directly instead of regenerating them.

If you want to avoid LLM-based generation, use `new_run.py` with explicit paths to the prepared example files.

### ✅ Using the Provided RTE Example Dataset

We have also provided an example dataset for `rte` to help users run the training pipeline without generating data from scratch.

Due to file upload size limitations, the full original RTE source dataset could not be uploaded to this repository. Instead, we provide a version of the RTE-based prepared data as a runnable example.

The provided  RTE example dataset is located at:

```text
/nli_arg_data/rte_lr0.8_gemma-e-4b-it-qat/
```

For this example dataset, we recommend using `new_run.py`.

Recommended command:

```bash
python new_run.py
```

Recommended training path in `new_run.py`:

```python
'train_data_path': '/nli_arg_data/rte_lr0.8_gemma-e-4b-it-qat/augmented_data.json'
```

Recommended validation path:

```python
'val_data_path': '/nli_arg_data/rte_lr0.8_gemma-e-4b-it-qat/validation.json'
```

If you prefer to use `augmented_data.json` as the training file, replace the training path with:

```python
'train_data_path': '/nli_arg_data/rte_lr0.8_gemma-e-4b-it-qat/augmented_data.json'
```

Using `new_run.py` with `use_augmented_data=False` directly loads these local files and avoids triggering LLM-based data generation.

Before running, please confirm that the paths in `new_run.py` actually match your local repository structure. Some paths in the original scripts may be hardcoded as absolute paths, such as `/home/ubuntu/...`, and should be changed to your own local paths or relative paths.

If you still want to use `NLImain.py`, make sure `NLIdataset.py` points to the correct data root:

```python
DEFAULT_AUG_ROOT = "/nli_arg_data"
```

Then run:

```bash
python NLImain.py --dataset_name rte --use_data_augmentation --labeled_ratio 0.8 --llm_model_name gemma-e-4b-it-qat
```

When the required JSON files already exist under `/nli_arg_data/rte_lr0.8_gemma-e-4b-it-qat/`, the code should load the prepared data directly instead of triggering LLM-based generation.

## 📂 Directory Organization

To ensure the code runs correctly, please organize your files as follows or update the paths in the code accordingly.

### Recommended Structure

```text
MetTrain/
├── NLImain.py
├── new_run.py
├── MRAugment.py
├── ... (other python scripts)
├── nli_dataset_cache/      <-- Cache for Hugging Face datasets
├── nli_arg_data/           <-- Provided  RTE example data
│   └── rte_lr0.8_gemma-e-4b-it-qat/
│       ├── augmented_data.json
│       ├── augmented_data.json
│       ├── labeled_data.json
│       ├── test.json
│       ├── unlabeled_data.json
│       └── validation.json
├── experiments/            <-- Experiment outputs generated by new_run.py
├── nli_saved_model/        <-- Default path for saving trained models
└── nli_model_cache/        <-- Default path for HuggingFace models
```

### ⚠️ Important Configuration Note

Currently, some paths may be hardcoded in the code. Before running experiments, please check the following.

If you use `new_run.py`, update the paths in the `config`, `train_datasets`, and validation dataset list sections:

```python
'train_data_path': '/nli_arg_data/rte_lr0.8_gemma-e-4b-it-qat/augmented_data.json'
'val_data_path': '/nli_arg_data/rte_lr0.8_gemma-e-4b-it-qat/validation.json'
'augmented_dir': '/nli_arg_data/rte_lr0.8_gemma-e-4b-it-qat/'
```

If the original code contains absolute paths such as:

```python
'/home/ubuntu/MTrain/MTrainNLI/nli_aug_data/...'
```

please replace them with paths that match your own local environment, for example:

```python
'/nli_arg_data/rte_lr0.8_gemma-e-4b-it-qat/validation.json'
```

If you use `NLImain.py`, open `NLIdataset.py` and modify the following variables to match your local environment:

```python
# In NLIdataset.py

# Change this to your desired path for storing augmented/prepared data.
# For the provided  RTE example dataset, use:
DEFAULT_AUG_ROOT = "/nli_arg_data"

# Change this to your desired path for caching datasets
cache_dir = "./nli_dataset_cache"
```

## Dataset JSON Format and Location

If you use local prepared data, dataset files should be stored in **JSON format** under the augmented-data directory expected by the training code.

For the provided  RTE example dataset, the root directory should be:

```text
/nli_arg_data/
```

Expected subdirectory naming rule:

```text
{dataset_name}_lr{labeled_ratio}_{llm_model_name}
```

The provided  RTE example follows this structure:

```text
/nli_arg_data/rte_lr0.8_gemma-e-4b-it-qat/
```

Expected JSON files may include:

* `augmented_data.json`
* `augmented_data.json`
* `labeled_data.json`
* `test.json`
* `unlabeled_data.json`
* `validation.json`

### Required Fields

The training and evaluation data should contain at least the following fields:

* `premise`
* `hypothesis`
* `label`

For augmented or metamorphic training, additional fields may be used:

* `pair_id`
* `is_syn`
* `mr_type`
* `weight`

`new_run.py` will automatically load the specified JSON or JSONL file and convert it into a Hugging Face `Dataset`.

## Model Storage

* **Pre-trained Models**: By default, the code uses Hugging Face's Model which downloads models to `nli_model_cache`. You can also specify a local path using `--model_name` or the `model_name` field in `new_run.py`.

## 📂 Project Structure

* `NLImain.py`: Original command-line entry point, handling argument parsing and the default training flow.
* `new_run.py`: Recommended configuration-based experiment runner, especially for prepared JSON or JSONL datasets.
* `MRAugment.py`: **Core file**, implementing various LLM-based metamorphic relation generation logic.
* `NLItrain.py` / `NLI_train_stage.py`: Contains training loops and specific training strategies, e.g., metamorphic relation loss functions.
* `NLI_train_one_epoch.py`: Contains one-epoch training functions used by `new_run.py`, including CE training, dual consistency training, multi-signal training, and R-Drop-style training.
* `NLIdataset.py`: Dataset loading and preprocessing.
* `NLImodel.py`: Model definition.
* `NLItest.py`: Evaluation utilities for NLI tasks.
* `LMstudio.py`: Interface responsible for communicating with the LLM backend.
* `seed_checker.py`: Utilities for random seed setting and reproducibility checking.
* `metamorphic_training_usage.py`: Example showing how to integrate metamorphic training into the code.

## 📝 Citation

If this code is helpful for your research, please cite our work after the paper is published. (To be added)
