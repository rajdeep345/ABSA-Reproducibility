# ABSA-Reproducibility
Codes and Datasets for our ECIR 2021 Paper: "Reproducibility, Replicability and Beyond: Assessing Production Readiness of Aspect Based Sentiment Analysis in the Wild"

## Setup instructions
* Create a conda environment using the requirements.txt file.
* Alternately, one can use the ABSA.yml extracted from our conda environment to exactly replicate the environment.
* Download and unzip the [GloVe embeddings](http://nlp.stanford.edu/data/glove.840B.300d.zip) into the current folder.

## Running experiments
```python
python grid_search.py
```

## Evaluating results
* Change directory to results/
```python
python process_results.py path [isHard]
```
path - Select one from [in_domain, contrast_logs, cross_domain, cross_domain_incremental]
isHard - [Default: 'False'] is set as 'True' only if you want to evaluate hard set results for in_domain experiments,  i.e.
```python
python process_results.py in_domain True
```

### Additional notes
* We run each experiment with 5 random seeds (1,2,3,4,5).
* Our experiments were run on a Tesla P100 PCIE, 16GB GPU and CUDA 10.1 and PyTorch 1.1.0.
* For the incremental cross domain experiments, the **--train\_dataset** argument can be set to **crossdomain\_indomain\_ratio**, for instance **Laptops_Restaurants_0.1** for evaluting the cross domain combination (Laptops - Train, Restaurants - Test).
