# EN3150 Assignment 03 — Resource-Constrained CNN for Edge Image Classification

**Group:** Rubix  
**Dataset:** Fashion-MNIST  
**Models:** Standard CNN (Model A), Lightweight CNN (Model B), MobileNetV2, and EfficientNetB0

## Project overview

This project compares classification accuracy and resource requirements for two custom convolutional neural networks and two pretrained architectures using Fashion-MNIST.

The dataset contains 70,000 grayscale images across 10 clothing classes. The project uses a 70/15/15 split:

- Training: 49,000 images
- Validation: 10,500 images
- Testing: 10,500 images

The custom CNNs use 28 × 28 grayscale images. MobileNetV2 and EfficientNetB0 use resized 64 × 64 three-channel images, with preprocessing appropriate to each architecture.

## Notebook execution order

Run the notebooks in this order:

1. `01_data_exploration.ipynb` — inspect the dataset and prepare the data.
2. `02_Model_A_Standard_CNN.ipynb` — train and evaluate the standard CNN.
3. `03_Model_B_Lightweight_CNN.ipynb` — train and evaluate the lightweight CNN and compare optimizers.
4. `04_SOTA_MobileNetV2.ipynb` — fine-tune and evaluate MobileNetV2.
5. `05_SOTA_EfficientNetB0.ipynb` — fine-tune and evaluate EfficientNetB0.

Run each notebook from top to bottom. If a notebook expects generated files from an earlier notebook, run the earlier notebook first and keep the expected relative folder structure.

## Environment setup

Python package versions were not consistently recorded in the exported notebooks. TensorFlow 2.21.0 is recorded in the Model B notebook; the requirements file therefore pins that version, while listing the main supporting packages without fixed versions.

Use a Python version supported by your installed TensorFlow release.

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
jupyter notebook
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
jupyter notebook
```

Open the notebooks in Jupyter and execute them in the order listed above.

## Expected outputs

The notebooks generate training-history plots, confusion matrices, evaluation metrics, and model/resource comparisons used in the report. Exact outputs depend on the runtime environment and execution settings.

## Reproducibility notes

- Use the same dataset split for all models.
- Keep the notebook execution order consistent.
- Record the Python version, TensorFlow version, hardware, and any changes to training settings when rerunning experiments.
- Training time can vary with hardware and software configuration.
- Do not compare saved model file sizes directly with estimated parameter-memory footprints; measure both models using the same saving and measurement procedure.

## Repository contents

The repository should contain these five notebooks, `README.md`, `requirements.txt`, and any assets or data-generation instructions required to run them. Avoid committing large generated files or datasets unless required by the course; document how to obtain or recreate them instead.
