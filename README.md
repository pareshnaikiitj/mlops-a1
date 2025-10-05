# mlops-a1

## Overview
This project is an MLOps pipeline that handles data preprocessing, model training, evaluation, and model persistence.  
It uses Python for the core implementation and is structured to follow best practices in modular ML development.

**Key Features:**
- Data loading, preprocessing, and feature extraction.
- Model training and evaluation with performance metrics.
- Modular structure for reusability (`misc.py`).
- Easy environment setup with `requirements.txt`.

---

## Project Structure

```text
mlops-a1/
│
├── README.md             # Project documentation
├── requirements.txt      # Python dependencies
├── misc.py               # Utility functions (data loading, preprocessing, etc.)
├── train.py              # Script for training and evaluating the model
├── train2.py             # Alternate training script
└── data/                 # Folder for datasets (raw/processed)