#  MSN-DTA
MSN-DTA: A multiscale node adaptive graph neural network for explainable drug-target
binding affinity prediction

##Data
You can download Davis and KIBA datasets from the following link:
http://staff.cs.utu.fi/~aatapa/data/DrugTarget/
Please use csv format

After downloading the data, please place it in the folder with the corresponding name, 
and then run processing.py for data processing to get the model input data

```python
python processing.py
```

## Requirements  
scikit-learn~=0.24.2
scipy=1.7.1
matplotlib==3.2.2  
pandas=1.3.4
torch_geometric==2.0.2
CairoSVG==2.5.2  
torch=1.12.1
tqdm=4.62.3 
networkx=2.6.3
numpy=1.20.3
rdkit=2021.09.2

## Pretrain
Since the model node information is obtained from pre training, 
the effect of pre training has a great impact on the final prediction performance of the model. 

For pre training, please set PreTrain in main.py to True.

```python
python main.py  --PreTrain = True
```
We provide a trained model, which you can import directly for testing

## Train model

After pre training, the model is saved in the save folder. Please set load_ Model_ Path is sandwiched in the pre training model
Please set PreTrain parameter to False to turn off pre training
```python
python main.py  --PreTrain = False
```

## Blind drug/target/drug-target test
Use split. py file to generate blind experiment dataset, Davis or KIBA can be set to obtain different blind datasets, 
and the random segmentation method can be adjusted through the "seed" parameter
```python
python split.py
```
Then you will receive the corresponding blind test dataset
