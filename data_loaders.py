import os
import numpy as np
import pandas as pd
import struct


numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
folder = "./data"

def shuffle_indexes(nrows, seed):
    np.random.seed(seed)
    index = np.arange(nrows)
    np.random.shuffle(index)
    return index

def load_mnist():
    with open("./data/train-images.idx3-ubyte",'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        mnist = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        mnist = (mnist.reshape((size, nrows * ncols)) / 255.).astype(np.float32)
    return mnist

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_cifar():
    cifar10 = np.empty(shape=(0,3072))
    for i in range(1,6):
        batch = unpickle("./data/cifar-10-batches-py/"+'data_batch_{}'.format(i))
        cifar10 = np.vstack((cifar10, batch[b'data'])) 
    cifar10 = (cifar10/255.).astype(np.float32)
    return cifar10

def load_dota2():
    name = "dota2+games+results/dota2Train.csv"
    path = os.path.join(folder, name)
    df = pd.read_csv(path, header=None)
    X = df.values.astype(np.float32)
    return X

def load_monitor_gas():
    name = "single+elder+home+monitoring+gas+and+position/DB elder monitoring/database_gas.csv"
    path = os.path.join(folder, name)
    df = pd.read_csv(path).select_dtypes(include=numerics)
    X = df.values.astype(np.float32)
    return X

def load_covertype():
    name = "covertype/covtype.data/covtype.data"
    path = os.path.join(folder, name)
    df = pd.read_csv(path)
    X = df.values.astype(np.float32)
    return X

def load_drybean():
    name = "dry+bean+dataset/DryBeanDataset/Dry_Bean_Dataset.xlsx"
    path = os.path.join(folder, name)
    df = pd.read_excel(path).select_dtypes(include=numerics)
    X = df.values.astype(np.float32)
    return X

def load_abalone():
    name = "abalone/abalone.data"
    path = os.path.join(folder, name)
    df = pd.read_csv(path).select_dtypes(include=numerics)
    X = df.values.astype(np.float32)
    return X

def load_bankruptcy():
    name = "taiwanese+bankruptcy+prediction/data.csv"
    path = os.path.join(folder, name)
    df = pd.read_csv(path).select_dtypes(include=numerics)
    X = df.values.astype(np.float32)
    return X

def load_mapping():
    name = "crowdsourced+mapping/training.csv"
    path = os.path.join(folder, name)
    df = pd.read_csv(path).select_dtypes(include=numerics)
    X = df.values.astype(np.float32)
    return X

def load_letter():
    name = "letter+recognition/letter-recognition.data"
    path = os.path.join(folder, name)
    df = pd.read_csv(path).select_dtypes(include=numerics)
    X = df.values.astype(np.float32)
    return X


datasets_dict = dict(
    dota2 = load_dota2,
    monitor_gas = load_monitor_gas,
    covertype = load_covertype,
    drybean = load_drybean,
    abalone = load_abalone,
    bankruptcy = load_bankruptcy,
    letter = load_letter,
    mapping = load_mapping,
)

def load_data(dataset, seed, n):
    caso = dataset.lower()
    if caso=='cifar':
        if n is None:
            n = 50000
        nrows = 50000
        shuffled_idxs = shuffle_indexes(nrows=nrows, seed=seed)
        return load_cifar()[shuffled_idxs][:n], shuffled_idxs[:n]
    elif caso == 'mnist':
        if n is None:
            n = 60000
        nrows = 60000
        shuffled_idxs = shuffle_indexes(nrows=nrows, seed=seed)
        return load_mnist()[shuffled_idxs][:n], shuffled_idxs[:n]
    else:
        data = datasets_dict[dataset]()
        print("\n Dataset %s - Shape: %s"%(dataset, str(data.shape)))
        return data, np.arange(len(data))
