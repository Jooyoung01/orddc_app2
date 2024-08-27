
# Optimized Road Damage Detection Challenge (ORDDC'2024)
## *The Models will be automatically downloaded, when you run the script.*
## Setup

### 0. Unzip orddc_2024.zip

```bash
unzip orddc_2024.zip -d ./
cd orddc_2024
```

### 1. Create and activate the conda environment

To create and activate the required conda environment, use the following commands:

```bash
conda create -n orddc_env python=3.10 -y
conda activate orddc_env
```

### 2. Install the required packages

After activating the environment, install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

1. Run the inference script using the following command:

```bash
python inference_script.py model.yaml ./images ./predictions.csv
```

### Positional Arguments:

- **model_file**: model YAML file name, including the directory name
- **source_path**: Path to the directory containing images for inference
- **output_csv_file**: output CSV file name including directory name
