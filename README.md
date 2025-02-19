# Condition Monitoring of Railway Vehicle Doors Using Domain Adaptation with Probabilistic Embedding
Testing code for [https://chains.dcollection.net/srch/srchDetail/200000859298?searchWhere1=all&insCode=243010&searchKeyWord1=%EA%B0%95%EB%8F%99%EC%A0%9C&query=%28ins_code%3A243010%29+AND++%2B%28%28all%3A%EA%B0%95%EB%8F%99%EC%A0%9C%29%29&navigationSize=10&start=0&pageSize=10&searthTotalPage=0&rows=10&ajax=false&pageNum=1&searchText=%5B%EC%A0%84%EC%B2%B4%3A%3Cspan+class%3D%22point1%22%3E%EA%B0%95%EB%8F%99%EC%A0%9C%3C%2Fspan%3E%5D&sortField=score&searchTotalCount=0&sortDir=desc]

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
