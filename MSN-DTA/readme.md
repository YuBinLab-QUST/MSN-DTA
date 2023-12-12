#  MSN-DTA
MSN-DTA: A multiscale node adaptive graph neural network for explainable drug-target
binding affinity prediction

## Requirement

torch=1.12.1
scikit-learn~=0.24.2
scipy=1.7.1
tqdm=4.62.3
pyg=2.0.2
numpy=1.20.3
rdkit=2021.09.2
networkx=2.6.3
pandas=1.3.4

##Data
You can download Davis and KIBA datasets from the following link:
http://staff.cs.utu.fi/~aatapa/data/DrugTarget/

Please use csv format

## Pretrain
Since the model node information is obtained from pre training, 
the effect of pre training has a great impact on the final prediction performance of the model. 
Please conduct pre training several times to get the optimal result.
which can be achieved by running main_ Pretraining. py for pre training

You can modify the hyperparameter customize your own training style with various parameters in pretraining. py
```python
python main_pretraining.py
```
## Train model
After pre training, use main.py to train the MSN-DTA model. 
The model needs to set the pre training file import path, and the model evaluation parameters should be set in the utils.py file
```python
python main.py
```

## Blind drug/target/drug-target test
Use split. py file to generate blind experiment dataset, Davis or KIBA can be set to obtain different blind datasets, 
and the random segmentation method can be adjusted through the "seed" parameter
```python
python split.py
```
Then you will receive the corresponding blind test dataset