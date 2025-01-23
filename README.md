# OneBatchPAM

This repository provides the results of the experiements conducted on MNIST, CIFAR10 and 8 UCI datasets.

Please clone the repository and install the dependencies with:
```
conda create -n onebatch-env python=3.11
pip install -r requirements.txt
```
Then build the onebatchpam algorithm with:
```
python setup.py build_ext --inplace
```
To run the experiments for the config file `configs/expe_large_scale.yml` use the following command line:
```
python script.py --config expe_large_scale
```

## Datasets

The 10 following datasets are used:
 - ``abalone`` Shape: (4176, 8) [[description](https://archive.ics.uci.edu/dataset/1/abalone)] [[download](https://archive.ics.uci.edu/static/public/1/abalone.zip)] 
 - ``bankruptcy`` Shape: (6819, 96) [[description](https://archive.ics.uci.edu/dataset/572/taiwanese+bankruptcy+prediction)] [[download](https://archive.ics.uci.edu/static/public/572/taiwanese+bankruptcy+prediction.zip)]
 - ``mapping`` Shape: (10545 28) [[description](https://archive.ics.uci.edu/dataset/400/crowdsourced+mapping)]
 - ``drybean`` Shape: (13611, 16) [[description](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset)] [[download](https://archive.ics.uci.edu/static/public/602/dry+bean+dataset.zip)]
 - ``letter`` Shape: (19999, 16) [[description](https://archive.ics.uci.edu/dataset/59/letter+recognition)] [[download](https://archive.ics.uci.edu/static/public/59/letter+recognition.zip)]
 - ``cifar`` Shape (50000, 3072) [[description](https://www.cs.toronto.edu/~kriz/cifar.html)]
 - ``mnist`` Shape (60000, 784) [[description](https://yann.lecun.com/exdb/mnist/)]
 - ``dota2`` Shape: (92650, 117) [[description](https://archive.ics.uci.edu/dataset/367/dota2+games+results
)] [[download](https://archive.ics.uci.edu/static/public/367/dota2+games+results.zip)] 
 - ``monitor_gas`` Shape: (416153, 9) [[description](https://archive.ics.uci.edu/dataset/799/single+elder+home+monitoring+gas+and+position)] [[download](https://archive.ics.uci.edu/static/public/799/single+elder+home+monitoring+gas+and+position.zip)]
 - ``covertype`` Shape: (581011, 55) [[description](https://archive.ics.uci.edu/dataset/31/covertype)] [[download](https://archive.ics.uci.edu/static/public/31/covertype.zip)]

To reproduce the experiements, please download the datasets and unzip them in a ``./data/`` folder.