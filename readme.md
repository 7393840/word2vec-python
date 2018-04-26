# word2vec-python

A word2vec python interface, using cpp extensions. 

### Install dependency

```
apt install gcc g++ make python3 python3-pip libboost-all-dev
```

### Compile

Set proper PYTHONINCLUDEPATH, PYTHONLIB, BOOSTPYTHONLIB, BOOSTNUMPYLIB. For example,  

```
PYTHONINCLUDEPATH = /usr/include/python3.6m
PYTHONLIB = python3.6m
BOOSTPYTHONLIB = boost_python3
BOOSTNUMPYLIB = boost_numpy3
```

Then

```
make
```

### usage

```
python word2vec.py TRAIN_FILE VEC_OUTPUT_FILE INDEX_OUTPUT_FILE
```