# ✍️ Handwritten Digit Recognition using KNN

[![Python](https://img.shields.io/badge/Python-v3.8+-yellow.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-v1.2x-013243.svg)](https://numpy.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-v1.x-orange.svg)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)

This repository contains a **custom K-Nearest Neighbors (KNN) classification engine** designed from scratch in Python to identify handwritten digits. Rather than relying on high-level library models, the custom classifier implements the mathematical formulations of Euclidean distance metrics and neighbor consensus voting manually.

---

## 🔬 Core Implementation Details

The core algorithm is built using standard matrix operations:
1. **Mathematical Distance Calculation (Euclidean Metric):**
   $$d(x^{(1)}, x^{(2)}) = \sqrt{\sum_{i=1}^n (x_i^{(1)} - x_i^{(2)})^2}$$
   Implemented manually using vectorization:
   ```python
   def euclidean_distance(x1, x2):
       return np.sqrt(np.sum((x1 - x2) ** 2))
   ```
2. **Nearest Neighbors Query Engine:**
   * Compute parent distances to every item in the training registry.
   * Sort candidates in ascending order of distance.
   * Pick the top $K$ neighbors.
3. **Consensus Majority Voting:**
   * Select most frequent classification label within top $k$ candidates:
   ```python
   unique, counts = np.unique(labels, return_counts=True)
   prediction = unique[counts.argmax()]
   ```

---

## 📂 Project Structure

```
project-hand-written/
├── -Handwritten-Digit-Recognition-using-KNN-algorithm-main/
│   ├── Handwritten Digit Recognition using KNN algorithm.ipynb  # Interactive walkthrough notebook
│   ├── run_knn.py                                               # Python pipeline execution script
│   ├── requirements.txt                                         # Required pip dependencies
│   ├── example_digit.png                                        # Exported sample prediction image
│   ├── mnist_train.rar                                          # Compressed MNIST training set
│   └── mnist_test.rar                                           # Compressed MNIST test set
└── README.md                                                    # Main repository documentation
```

---

## 🚀 Getting Started

To run the project locally on your machine, follow these instructions:

### 1. Installation
Navigate to the project subdirectory:
```bash
cd project-hand-written/-Handwritten-Digit-Recognition-using-KNN-algorithm-main
```

Install the dependencies:
```bash
pip install -r requirements.txt
```

### 2. Running the script
To run the automated model script:
```bash
python run_knn.py
```
This loads Scikit-learn's digits dataset, splits it, normalizes pixels, calculates accuracy, and outputs `example_digit.png` highlighting predictions.

### 3. Notebook Walkthrough
To run the interactive notebook showing MNIST graphs:
```bash
jupyter notebook
```
Open the `.ipynb` file to run cell-by-cell visualizations.
