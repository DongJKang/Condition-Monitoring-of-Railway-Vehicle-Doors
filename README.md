# Condition Monitoring of Railway Vehicle Doors Using Domain Adaptation with Probabilistic Embedding
PyTorch implementation for validation

## Getting started

### Installation
Install library versions that are compatible with your environment.
```bash
git clone https://github.com/DongJKang/Condition-Monitoring-of-Railway-Vehicle-Doors.git
cd Condition-Monitoring-of-Railway-Vehicle-Doors
conda create -n door python=3.9
conda activate door
pip install -r requirements.txt

```

### Recommended configuration

```
python=3.9
numpy=1.26.4
sklearn=1.3.1
pytorch=1.12.1
```

### Usages
Running the code below will execute the tests:
```
python main.py
```

## Test results

Task1: T1 -> T2
| PE            | UDA            | test f1-score               |
| ------------- | -------------- | ---------------------- |
| ✗             | ✗             | src: 97.28 tgt: 91.14  |
| ✗             | ✓ (mcd)       | src: 96.49 tgt: 96.02  |
| ✓ (Gaussian)  | ✗             | src: 96.49 tgt: 93.78  |
| ✓ (Gaussian)  | ✓ (mcd)       | src: 98.84 tgt: **96.40**  |


Task2: T2b -> F
| PE            | UDA            | test f1-score               |
| ------------- | -------------- | ---------------------- |
| ✗             | ✗             | src: 97.22 tgt: 89.74  |
| ✗             | ✓ (adda)      | src: 63.16 tgt: 92.47  |
| ✓ (Gaussian)  | ✗             | src: 97.14 tgt: 89.74  |
| ✓ (Gaussian)  | ✓ (adda)      | src: 66.67 tgt: **97.73**  |