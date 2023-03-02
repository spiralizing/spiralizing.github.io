+++
title = "Predicting wine quality using chemical properties"
hascode = true
date = Date(2023, 1, 5)
rss = "Predicting wine quality with NN"

tags = ["ANN", "data science", "wine quality","Neural Networks"]
+++
As a food and drink enthusiast I've always wondered if we can use statistics to predict flavor profile or quality/taste in food. I know there is a whole research field when it comes to [food science](https://en.wikipedia.org/wiki/Food_science) that explores these kind of questions, but I wanted to see if I could try it on my own.

Particularly, when I talk to people about drinks is usually about beer or whisky, but when the conversation goes toward wine I often feel lost, I don't think I have the sophisticated palate that other people have. So when I realized that there is a [data set of wine quality](https://archive-beta.ics.uci.edu/dataset/186/wine+quality), first published on a [paper](https://www.sciencedirect.com/science/article/abs/pii/S0167923609001377) that includes different chemical properties and a quality score provided by wine experts. I decided to check it out and to see if I could use statistics (machine learning) to explore and try to answer questions I've wondered for a while, like: **Can we predict the quality/taste of a wine knowing its chemical properties?**

This entry has the following sections:

\toc 

## Loading imports and data
First, we load some libraries we are going to use for this mini-project:
```python
#importing libraries
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import sklearn.metrics as skm

import numpy as np
import scipy.stats as stats

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from IPython import display
display.set_matplotlib_formats('svg')
```

```python
#getting the dataset - there are 2 sets: red wines and white wines
url_white = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
url_red = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
rwine_data = pd.read_csv(url_red, sep=';')
wwine_data = pd.read_csv(url_white, sep=';')
```
We assign labels for type of wine on each data frame and then concatenate them to have all the information in one single data frame:

```python 
#assigning labels
rwine_data['wine type'] = 1
wwine_data['wine type'] = 0

#concatenating data frames
allwine_data = pd.concat([rwine_data, wwine_data], axis=0, ignore_index=True)
```
it is fairly common that the data sets contain unbalanced and not normalized data so it won't hurt if we look at its distribution: 

```python
#plot data 
fig, ax = plt.subplots(1, figsize=(17,4))
ax = sns.boxplot(data=allwine_data)
ax.set_xticklabels(allwine_data.columns,rotation=45)
```
~~~
<div class="container">

    <img class="center" src="/assets/chem_distro1.svg" width="500" height="350">

</div>
~~~
the distributions of the wine properties have different scales so it will be convenient to normalize them

```python
# we drop the columns quality and wine type since those don't need normalization.
normed_alldata = allwine_data.drop(['quality', 'wine type'], axis=1)

for col in normed_alldata.columns:
    #getting mean and standard deviation...
    col_mean = np.mean(normed_alldata[col])
    col_std = np.std(normed_alldata[col], ddof=1)
    #normalizing data
    normed_alldata[col] = (normed_alldata[col] - col_mean) / col_std

```
now we check that the data is normalized by plotting the new distributions

```python
#plot the normalized data
fig, ax = plt.subplots(1, figsize=(17,4))
ax = sns.boxplot(data=normed_alldata)
ax.set_xticklabels(normed_alldata.columns,rotation=45)
```
~~~
<div class="container">

    <img class="center" src="/assets/chem_distronorm.svg" width="500" height="350">

</div>
~~~

## Predicting wine type

Before trying to predict wine quality, a good exercise would be to see how accurately we can predict other properties with the data we already have. The type of wine could be the simplest one so we will start with that one.

Let's see how many points we have of each type of wine: 

```python
#plotting the counts for each type of wine
plt.hist(allwine_data['wine type'].values)
plt.xticks([0,1], labels=['White','Red'])
```
~~~
<div class="container">

    <img class="center" src="/assets/count_winetype.svg" width="350" height="350">

</div>
~~~
there is a disproportion of data points, if we train a model with this data it will be likely to be biased towards predicting white wine (better trained for predicting white wine), so we need to use a balanced dataset.

To balance the dataset we need to create more data points or reduce their number, for simplicity we are going to perform the second option. We need to randomly select a subset of points from the white wine group of data points 

```python
# making a *randomized* selection of white wine data
white_ix = np.where(allwine_data['wine type']==0)[0]
red_ix = np.where(allwine_data['wine type']==1)[0]

#this are the indexes we are going to choose for our dataset
ix_selec = np.concatenate([np.random.choice(white_ix, len(red_ix)), red_ix])
```

now we can define the variables for our machine learning model

```python
#defining the data and target
normed_allselect = normed_alldata.iloc[ix_selec]
wine_target = allwine_data['wine type'].values[ix_selec]
```
in Pytorch an easy way to pass the data to the model is through loaders, so we can define a function to build the loaders and make it easier if we decide to make more experiments with different datasets 

```python
# function that will help us to build the data loaders for PyTorch
def get_loaders(data, target, test_size, batch_size):
    #first converting to torch format
    Tdata = torch.tensor(data.values).float()
    Ttarget = torch.tensor(target).float()[:, None]
    
    #split the data
    train_data, test_data, train_labels, test_labels = train_test_split(Tdata, Ttarget, test_size=test_size)
    
    #creating datasets for the loaders
    train_data = TensorDataset(train_data, train_labels)
    test_data = TensorDataset(test_data, test_labels)
    
    #creating loaders, dropping last to have equal size loaders.
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_data, batch_size=test_data.tensors[0].shape[0])
    
    return train_loader, test_loader
```
and we use the function we just defined to build the loaders with our data, using 80% of the data for training and the rest for testing with batches of size `batch_size = 64`

```python
train_loader, test_loader = get_loaders(
    normed_allselect, wine_target, test_size=0.2, batch_size=64)

```

Now we can think about our [Artificial Neural Network (ANN)](https://en.wikipedia.org/wiki/Artificial_neural_network) model, I decided to select a model flexible enough to be used for our different experiments. The model is _fully connected_ (FCNN) which means that all nodes from neighbor layers are connected to each other with the exception of self-loops and nodes from the same layer. It has an input layer, four hidden layers and an output layer, and we can visualize the architecture in the next figure

~~~
<div class="container">

    <img class="center" src="/assets/nn.svg" width="600" height="600">

</div>
~~~

to build this model we can use the Torch base NN class:

```python
#defining our class for this ANN model fully connected with batch normalization
class ANN_wine(nn.Module):
    def __init__(self, n_input):  # initiating class, setting number of input nodes as variable
        super().__init__()

        #intput for the n_input features to 16 nodes
        self.input = nn.Linear(n_input, 16)

        #First layer
        self.fc1 = nn.Linear(16, 32)
        self.bnorm1 = nn.BatchNorm1d(16)
        #second layer
        self.fc2 = nn.Linear(32, 20)
        self.bnorm2 = nn.BatchNorm1d(32)
        #third layer
        self.fc3 = nn.Linear(20, 16)
        self.bnorm3 = nn.BatchNorm1d(20)
        #the output
        self.output = nn.Linear(16, 1)

    #defining forward pass with ReLu activation functions
    def forward(self, x):  
        #input forward
        x = F.relu(self.input(x))

        #forward of layer 1 with batch normalization
        x = self.bnorm1(x)
        x = F.relu(self.fc1(x))
        #forward of layer 2 with  batch normalization
        x = self.bnorm2(x)
        x = F.relu(self.fc2(x))
        #forward of layer 3 with  batch normalization
        x = self.bnorm3(x)
        x = F.relu(self.fc3(x))

        return self.output(x)

```
before training our model we need to make sure that the data loaders and the model have the same format, to check this we just need to pass one batch from the loader through the model

```python

#Loading a batch from the training dataset
X, y = next(iter(train_loader))
print(X.shape)
print(y.shape)

#we initialize our model with 11 input features
ANN_winetype = ANN_wine(11)
#pass the model
y_hat = ANN_winetype(X)
print(y_hat.shape)

```
```plaintext
torch.Size([64, 11])
torch.Size([64, 1])
torch.Size([64, 1])
```
it seems that everything is in order. Now we need to train the model, we define a function to train the model that uses Binary [Cross-entropy](https://en.wikipedia.org/wiki/Cross_entropy) as loss function and the [Adam optimizer](https://optimization.cbe.cornell.edu/index.php?title=Adam#:~:text=Adam%20optimizer%20is%20the%20extended,was%20first%20introduced%20in%202014.):  

```python
def train_model(model,train_loader, test_loader, n_epochs, learn_rate):
    #loss function: binary cross entropy.
    lossfun = nn.BCEWithLogitsLoss()
    #defining the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
    
    #now we initialize the losses and accuracy
    losses = torch.zeros(n_epochs)
    train_acc = []
    test_acc = []
    for epoch_i in range(n_epochs):
        #using the model in training mode
        model.train()
        batch_acc = []
        batch_loss = []
        
        #Now we loop over the batches
        for X, y in train_loader:
            #forward
            y_hat = model(X)
            loss = lossfun(y_hat, y)
            
            #back propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            #storing batch loss and accuracy
            batch_loss.append(loss.item())
            batch_acc.append( 100*torch.mean( ((y_hat>0) == y).float()).item())
        
        #saving epoch's loss and accuracy 
        train_acc.append(np.mean(batch_acc))
        losses[epoch_i] = np.mean(batch_loss)
        
        #now we evaluate the model with the test data
        #switch to evaluation mode
        model.eval()
        X, y = next(iter(test_loader))
        
        #forward
        with torch.no_grad():
            y_hat = model(X)
        #accuracy for test data    
        test_acc.append( 100*torch.mean(((y_hat>0) == y).float()).item())
        
    return train_acc, test_acc, losses
```
now we can train our model and plot the loss and accuracy over epochs
```python
train_acc, test_acc, losses = train_model(ANN_winetype,
                                          train_loader,
                                          test_loader,
                                          n_epochs=200,
                                          learn_rate=0.001
                                          )

```

```python
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

ax[0].plot(losses, 'k-')
ax[0].set_xlabel('Epoch')
ax[0].set_ylabel('Loss')
ax[0].set_title('Loss per Epoch')
ax[0].grid()

ax[1].plot(train_acc)
ax[1].plot(test_acc)
ax[1].set_xlabel('Epoch')
ax[1].set_ylabel('Accuracy (%)')
ax[1].set_title(f'Final accuracy: {test_acc[-1]:.2f}%')
ax[1].grid()
```

~~~
<div class="container">

    <img class="center" src="/assets/traintest_winetype.svg" width="500" height="400">

</div>
~~~
From these plots we can argue that our model was too elaborated for this task, it took few epochs to reach a maximum in accuracy. To evaluate the performance of our model we import some metrics from `sklearn` like the [confusion matrix](https://en.wikipedia.org/wiki/Confusion_matrix) and a classification report that includes values for: [precision, recall](https://en.wikipedia.org/wiki/Precision_and_recall) and [f1-score](https://en.wikipedia.org/wiki/F-score).  
```python
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

X, y = next(iter(test_loader))
y_hat = ANN_winetype(X)
pred_wine = (y_hat > 0).detach().numpy()

CM_winetype = confusion_matrix(y, pred_wine)
```
plotting the confusion matrix:

```python
plt.rcParams.update({'font.size': 15})
plt.figure(figsize=(8, 8))
plt.imshow(CM_winetype, 'Oranges')
plt.xticks([0, 1], ['White', 'Red'])
plt.yticks([0, 1], ['White', 'Red'])
plt.xlabel('Predicted type')
plt.ylabel('True type')

plt.text(0, 0, f'True negatives:\n{CM_winetype[0,0]}', ha='center', va='center')
plt.text(
    0, 1, f'False negatives:\n{CM_winetype[1,0]}', ha='center', va='center')
plt.text(1, 1, f'True positives:\n{CM_winetype[1,1]}', ha='center', va='center')
plt.text(
    1, 0, f'False positives:\n{CM_winetype[0,1]}', ha='center', va='center')

plt.show()
```
~~~
<div class="container">

    <img class="center" src="/assets/cm_winetype.svg" width="500" height="450">

</div>
~~~
since we didn't get any false positives/negatives the precision and recall have _perfect_ values and we expect to have a perfect f1-score since this is computed from precision and recall, to confirm this we print the classification report
```python
cr = classification_report(y, pred_wine)
print(cr)
```
```plaintext
        ---------        precision    recall  f1-score   support

         0.0       1.00      1.00      1.00       326
         1.0       1.00      1.00      1.00       314

    accuracy                           1.00       640
   macro avg       1.00      1.00      1.00       640
weighted avg       1.00      1.00      1.00       640
```
But, why is that this model performs so well, my first guess is that chemical properties between white and red wines are **very** different. Instead of comparing all the properties in our data, I decided to do a small search on Google and chose the chemical properties that contribute the most to flavor profile and found [here](https://www.sciencehistory.org/distillations/scientia-vitis-decanting-the-chemistry-of-wine-flavor) that the main component is **sugar** (residual sugar), followed by acid and tannins.

So we will going to drop some of the properties from the full dataset
```python
#selecting chemical properties that contribute to flavor
chem_prop = normed_allselect.columns.drop(
    ['free sulfur dioxide', 'total sulfur dioxide', 'sulphates', 'density', 'chlorides']).values
chem_prop

```
and then we can see how the remaining distributions differ with the type of wine
```python
plt.rcParams.update({'font.size': 10})

fig, ax = plt.subplots(2, 3, figsize=(14, 9))
prop_c = 0  # counter

for i in range(2):
    for j in range(3):
        sns.boxplot(data=allwine_data, x='wine type', y = chem_prop[prop_c], ax=ax[i,j])
        ax[i,j].set_xticklabels(['white','red'])
        prop_c += 1

plt.show()
```
~~~
<div class="container">

    <img class="center" src="/assets/chem_props.svg" width="600" height="600">

</div>
~~~
as we can see, both types of wine have very different chemical properties, white wines tend to be more sugary on average, on the other hand red wines have larger average values of fixed and volatile acidity. It would not be surprising if by using simpler models (linear regression, decision tree, statistical tests) we can achieve the same type of result.

## Can we predict the value of a chemical property?

Now let's see if we can predict chemical properties related to flavor profile, since both type of wine have different chemical properties we are going to use only the data of one type of wine. We will work with the white wine dataset since it is the one with the largest amount of data points.

We normalize the data and construct loaders

```python
#we are going to normalize wine properties except for the quality (or target)
normed_wdata = wwine_data.drop(['quality', 'wine type'], axis=1)

for col in normed_wdata.columns:
    #getting mean and standard deviation...
    col_mean = np.mean(normed_wdata[col])
    col_std = np.std(normed_wdata[col], ddof=1)
    #normalizing data 
    normed_wdata[col] = (normed_wdata[col] - col_mean) / col_std
```
we do the first property in the list: **Fixed Acidity**

```python
#the first chemical property in the list 
new_data = normed_wdata.drop(chem_prop[0], axis=1)
new_target = wwine_data[chem_prop[0]].values

#build loaders
train_loader, test_loader = get_loaders(new_data, new_target, test_size=0.2, batch_size=64)
```
now we need to define another training function, it should be different since we will be performing a **regression** for these values (real numbers) and we need a different loss function like [Mean Square Error (MSE)](https://en.wikipedia.org/wiki/Mean_squared_error). And as a quantity to test accuracy we are going to use [Pearson Correlation](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient) that indicates linear statistical dependencies between two variables.

```python
def train_model_2(model, train_loader, test_loader, n_epochs, learn_rate):
    #loss function: Mean Square Error - for regression.
    lossfun = nn.MSELoss()
    #defining the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

    #now we initialize the losses and accuracy
    losses = torch.zeros(n_epochs)
    train_acc = []
    test_acc = []
    for epoch_i in range(n_epochs):
        #using the model in training mode
        model.train()
        batch_acc = []
        batch_loss = []

        #Now we loop over the batches
        for X, y in train_loader:
            #forward
            y_hat = model(X)
            loss = lossfun(y_hat, y)

            #back propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #storing batch loss and accuracy
            batch_loss.append(loss.item())
            #computing pearson correlation for "accuracy"
            batch_cor = np.corrcoef(np.concatenate(np.array(y_hat.detach())),np.concatenate(np.array(y)))[0,1]
            batch_acc.append(batch_cor)

        #saving epoch's loss and accuracy
        train_acc.append(np.mean(batch_acc))
        losses[epoch_i] = np.mean(batch_loss)

        #now we evaluate the model with the test data
        #switch to evaluation mode
        model.eval()
        X, y = next(iter(test_loader))

        #forward
        with torch.no_grad():
            y_hat = model(X)
        #accuracy for test data
        test_cor = np.corrcoef(np.concatenate(
            np.array(y_hat.detach())), np.concatenate(np.array(y)))[0, 1]
        test_acc.append(test_cor)

    return train_acc, test_acc, losses
```

we initialize our model and make a test by passing one batch, we are using the same function (ANN architecture) but with different number of inputs this time (10 instead of 11):

```python
#defining a new model and passing data through it 
chemprop_model = ANN_wine(10)

X, y = next(iter(train_loader))
#evaluating
y_hat = chemprop_model(X)
print(X.shape)
print(y.shape)
print(y_hat.shape)

```
```plaintext
torch.Size([64, 10])
torch.Size([64, 1])
torch.Size([64, 1])
```


```python
m2_trainacc, m2_testacc, m2_losses = train_model_2(chemprop_model, 
    train_loader, 
    test_loader, 
    n_epochs=200, 
    learn_rate=0.001)
```
```python
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

ax[0].plot(m2_losses, 'k-')
ax[0].set_xlabel('Epoch')
ax[0].set_ylabel('Loss')
ax[0].set_title('Loss per Epoch')
ax[0].grid()

ax[1].plot(m2_trainacc)
ax[1].plot(m2_testacc)
ax[1].set_xlabel('Epoch')
ax[1].set_ylabel(r'Pearson Correlation Coefficient ($\rho$)')
ax[1].set_title('Final ' r'$\rho$' f': {m2_testacc[-1]:.2f}')
ax[1].grid()
```
~~~
<div class="container">

    <img class="center" src="/assets/traintest_fixedac.svg" width="500" height="400">

</div>
~~~
and evaluate the final performance from the training, with the correlation coefficient for the train and test data

```python
#loading all train and test data
train_data = train_loader.dataset.tensors
test_data = test_loader.dataset.tensors

#passing the data through the model to predict Fixed Acidity values
train_pred = chemprop_model(train_data[0])
test_pred = chemprop_model(test_data[0])

#computing pearson correlation coefficient of target and predicted values for train and test data
pcor_train = np.corrcoef(np.concatenate(np.array(train_pred.detach().numpy())),np.concatenate(train_data[1].detach().numpy()))[0,1]
pcor_test = np.corrcoef(np.concatenate(np.array(test_pred.detach().numpy())),np.concatenate(test_data[1].detach().numpy()))[0,1]  
```
plotting the estimations
```python
plt.figure(figsize=(7,7))
plt.plot(train_pred.detach(), train_data[1].detach(), 'o')
plt.plot(test_pred.detach(), test_data[1].detach(), 'o')
plt.grid()
plt.legend(('Train ' r'$\rho$' f': {pcor_train:.2f}', 'Test ' r'$\rho$' f': {pcor_test:.2f}'))
plt.title('Fixed Acidity')
#plt.xlim([0,60])
plt.xlabel('Predicted Value')
plt.ylabel('Real Value')
plt.axline((0, 0), slope=1, linestyle='--', color='k', label=None)
plt.show()
```
~~~
<div class="container">

    <img class="center" src="/assets/cor_fixedac.svg" width="500" height="450">

</div>
~~~
the dashed line represents the identity ($f(x) = x$), helping us as a reference since the closer the points are to the identity the more accurate the prediction is. As expected the model performs better with the training set, but the correlation coefficient of almost 90% in the test set indicates a good prediction.

We do the same analysis for each chemical property:
```python
#variables to store data
pcor_trains = np.zeros(6)
pcor_tests = np.zeros(6)
pred_trains = []
pred_tests = []
data_trains = []
data_tests = []

#looping over the chemical properties
for (nc,chemprop) in enumerate(chem_prop):
    #building data
    new_data = normed_data.drop(chemprop, axis=1)
    new_target = wwine_data[chemprop].values
    #loaders
    train_loader, test_loader = get_loaders(
        new_data, new_target, test_size=0.2, batch_size=64)
    #initializing model
    chemprop_model = ANN_wine(10)
    #training... 
    _ = train_model_2(
        chemprop_model, train_loader, test_loader, n_epochs=200, learn_rate=0.001)

    #computing final accuracy
    train_data = train_loader.dataset.tensors
    test_data = test_loader.dataset.tensors

    train_pred = chemprop_model(train_data[0])
    test_pred = chemprop_model(test_data[0])
    
    pcor_train = np.corrcoef(np.concatenate(np.array(train_pred.detach().numpy())),np.concatenate(train_data[1].detach().numpy()))[0,1]
    pcor_test = np.corrcoef(np.concatenate(np.array(test_pred.detach().numpy())),np.concatenate(test_data[1].detach().numpy()))[0,1]  

    pcor_trains[nc] = pcor_train
    pcor_tests[nc] = pcor_test
    
    pred_trains.append(train_pred)
    pred_tests.append(test_pred)
    data_trains.append(train_data[1])
    data_tests.append(test_data[1])

```
and plot the results

```python
fig, ax = plt.subplots(2, 3, figsize=(18, 12))
prop_c = 0 #counter

for i in range(2):
    for j in range(3):
        xl = [np.min(pred_trains[prop_c].detach().numpy()), np.max(pred_trains[prop_c].detach().numpy())]
        yl = [np.min(pred_trains[prop_c].detach().numpy()), np.max(pred_trains[prop_c].detach().numpy())]
        ax[i,j].plot(pred_trains[prop_c].detach(), data_trains[prop_c].detach(), 'o')
        ax[i,j].plot(pred_tests[prop_c].detach(), data_tests[prop_c].detach(), 'o')
        ax[i,j].grid()
        ax[i,j].set_xlim(xl)
        ax[i,j].set_ylim(yl)
        ax[i,j].legend(('Train ' r'$\rho$' f': {pcor_trains[prop_c]:.2f}',
                'Test ' r'$\rho$' f': {pcor_tests[prop_c]:.2f}'))
        ax[i,j].set_title(chem_prop[prop_c])
        #ax[i,j].xlim([0, 60])
        ax[i,j].set_xlabel('Predicted Value')
        ax[i,j].set_ylabel('Real Value')
        ax[i,j].axline((0, 0), slope=1, linestyle='--', color='k', label=None)
        prop_c += 1

```
~~~
<div class="container">

    <img class="center" src="/assets/regression_chemprops.svg" width="700" height="700">

</div>
~~~

almost all chemical properties can be predicted with a decent amount of precision, however if we check the pearson correlation values for the test data there are a couple of poor performances: _volatile acidity_ and _citric acid_.  


## Predicting wine quality
Finally, we will use the same model to predict wine quality, first let's check how the quality scores are distributed

```python
target = wwine_data['quality'].values

plt.hist(target)
plt.xlabel('Quality Score')
plt.ylabel('Count')
plt.show()
```
~~~
<div class="container">

    <img class="center" src="/assets/quality_whitedistros.svg" width="300" height="300">

</div>
~~~

there are too many categories and the data is not distributed uniformly, to simplify this we will make a binary quality score (good, not good) and then try to balance the data so we don't bias our training process.

First we binarize the data:

```python
#we set 0 = not good, 1 = good
boolean_target = np.array([0 if target[i] < 6 else 1 for i in range(len(target))] )

#plotting new distribution
plt.hist(boolean_target)
plt.xlabel('Quality Score')
plt.ylabel('Count')
plt.xticks([0,1], labels=['Not good', 'Good'])
plt.show()
```
~~~
<div class="container">

    <img class="center" src="/assets/binary_qualitycount.svg" width="300" height="300">

</div>
~~~
we can identify a clear unbalance between the two categories, this is the same problem we had before where we needed to select a subset of datapoints, we will do the same in this case

```python
#locating both categories
good_ix = np.where(boolean_target==1)[0]
bad_ix = np.where(boolean_target==0)[0]

#number of data points to be sampled
n_samps = len(bad_ix)

#selecting at random
# we choose at random n_samps for the 1 'quality' 
ix_selec = np.concatenate([np.random.choice(good_ix, n_samps), bad_ix])

# we use the selection for the data 
btarget_select = boolean_target[ix_selec]
normed_select = normed_wdata.iloc[ix_selec]
```

Now we can finally try to predict the quality of the wine, we build the loaders, initialize the model and pass data through it

```python
#building loaders
train_loader, test_loader = get_loaders(normed_select, 
    btarget_select, 
    test_size=0.2, 
    batch_size=64)

#initializing model
ANN_model = ANN_wine(11)

#passing data through the model
X,y = next(iter(train_loader))
print(X.shape)
print(y.shape)
y_hat = ANN_model(X)
print(y_hat.shape)
```
```plaintext
torch.Size([64, 11])
torch.Size([64, 1])
torch.Size([64, 1])
```

```python
train_acc, test_acc, losses = train_model(ANN_model, 
                                          train_loader, 
                                          test_loader,
                                          n_epochs=1000,
                                          learn_rate=0.001
                                          )

fig, ax = plt.subplots(1, 2, figsize=(12, 5))

ax[0].plot(losses, 'k-')
ax[0].set_xlabel('Epoch')
ax[0].set_ylabel('Loss')
ax[0].set_title('Loss per Epoch')
ax[0].grid()

ax[1].plot(train_acc)
ax[1].plot(test_acc)
ax[1].set_xlabel('Epoch')
ax[1].set_ylabel('Accuracy (%)')
ax[1].set_title(f'Final accuracy: {test_acc[-1]:.2f}%')
ax[1].grid()
```
~~~
<div class="container">

    <img class="center" src="/assets/traintest_whitewinequality.svg" width="500" height="400">

</div>
~~~

In this case, the accuracy is not as high as with the chemical properties or the type of wine, and it seems that we could still get lower loss and better training accuracy if we train for more epochs, however the lack of improvement on test accuracy tells us that it might not be necessary to train for longer.

Now let's se how good our ANN model is to predict the wine quality given by experts

```python

X, y = next(iter(test_loader))
y_hat = ANN_model(X)
pred_wwine = (y_hat > 0).detach().numpy()

#confusion matrix
c_matrix = confusion_matrix(y, pred_wwine)

plt.rcParams.update({'font.size': 15})
plt.figure(figsize=(8, 8))
plt.imshow(c_matrix, 'Blues')
plt.xticks([0,1], ['Bad','Good'])
plt.yticks([0,1],['Bad','Good'])
plt.xlabel('Predicted quality')
plt.ylabel('True quality')

plt.text(0, 0, f'True negatives:\n{c_matrix[0,0]}', ha='center', va='center')
plt.text(
    0, 1, f'False negatives:\n{c_matrix[1,0]}', ha='center', va='center')
plt.text(1, 1, f'True positives:\n{c_matrix[1,1]}', ha='center', va='center')
plt.text(
    1, 0, f'False positives:\n{c_matrix[0,1]}', ha='center', va='center')

plt.show()
```
~~~
<div class="container">

    <img class="center" src="/assets/cm_whitewinequality.svg" width="500" height="500">

</div>
~~~
```python 
cr_wwine = classification_report(y, pred_wwine)
print(cr_wwine)
```
```plaintext
---             precision    recall  f1-score   support

         0.0       0.83      0.79      0.81       327
         1.0       0.80      0.84      0.82       329

    accuracy                           0.82       656
   macro avg       0.82      0.82      0.82       656
weighted avg       0.82      0.82      0.82       656

```
These results are decent enough to think our model does capture some hidden rules in the dependency of the chemical properties with the quality of the wine, but they fall short if we compare them with previous results.

But, what about red wine? We do the same analysis for red wine

```python
#Normalizing data
normed_rdata = rwine_data.drop(['quality', 'wine type'], axis=1)

for col in normed_rdata.columns:
    #getting mean and standard deviation...
    col_mean = np.mean(normed_rdata[col])
    col_std = np.std(normed_rdata[col], ddof=1)
    #normalizing data
    normed_rdata[col] = (normed_rdata[col] - col_mean) / col_std

#binarizing target
target = rwine_data['quality'].values
boolean_target = np.array(
    [0 if target[i] < 6 else 1 for i in range(len(target))])

#plotting target distribution
plt.hist(boolean_target)
plt.xlabel('Quality Score')
plt.ylabel('Count')
plt.xticks([0, 1], labels=['Not good', 'Good'])
plt.show()
```
~~~
<div class="container">

    <img class="center" src="/assets/binary_redqualitycount.svg" width="300" height="300">

</div>
~~~
it seems that the data is slightly unbalanced, so for this time we will skip the balancing part and train and test our model

```python
#building loaders
train_loader, test_loader = get_loaders(
    normed_rdata, boolean_target, test_size=0.2, batch_size=64)

#checking that the loaders are correct
X, y = next(iter(train_loader))
print(X.shape)
print(y.shape)

#init model
ANN_model = ANN_wine(11)
#passing data
y_hat = ANN_model(X)
print(y_hat.shape)
```
```plaintext
torch.Size([64, 11])
torch.Size([64, 1])
torch.Size([64, 1])
```

```python
train_acc, test_acc, losses = train_model(ANN_model,
                                          train_loader,
                                          test_loader,
                                          n_epochs=1000,
                                          learn_rate=0.001
                                          )

X, y = next(iter(test_loader))
y_hat = ANN_model(X)
pred_rwine = (y_hat > 0).detach().numpy()

c_matrix = confusion_matrix(y, pred_rwine)
plt.rcParams.update({'font.size': 15})
plt.figure(figsize=(8, 8))
plt.imshow(c_matrix, 'Blues')
plt.xticks([0, 1], ['Bad', 'Good'])
plt.yticks([0, 1], ['Bad', 'Good'])
plt.xlabel('Predicted quality')
plt.ylabel('True quality')

plt.text(0, 0, f'True negatives:\n{c_matrix[0,0]}', ha='center', va='center')
plt.text(
    0, 1, f'False negatives:\n{c_matrix[1,0]}', ha='center', va='center')
plt.text(1, 1, f'True positives:\n{c_matrix[1,1]}', ha='center', va='center')
plt.text(
    1, 0, f'False positives:\n{c_matrix[0,1]}', ha='center', va='center')

plt.show()
```
~~~
<div class="container">

    <img class="center" src="/assets/cm_redwinequality.svg" width="500" height="500">

</div>
~~~

```python
cr_rwine = classification_report(y, pred_rwine)
print(cr_rwine)
```
```plaintext
 ----             precision    recall  f1-score   support

         0.0       0.67      0.75      0.71       130
         1.0       0.82      0.75      0.78       190

    accuracy                           0.75       320
   macro avg       0.74      0.75      0.75       320
weighted avg       0.76      0.75      0.75       320

```

our model performs slightly worse in predicting quality score for red wine, it could be related to the size of the dataset. But why is that we can have higher accuracy in predicting the amount of residual sugar?

this discrepancy could have different explanations: 

* Our model (approach) is not good enough
* Wine quality could be related to the acidity values we couldn't predict accurately
* Wine quality scores have a **subjective** component from the _experts_ 

it is not simple to know what exactly might be the issue, but we can ignore the first one since we are using the same model for each prediction, of course we could have a better model but that is not the point of this experiment, in the [original paper](https://www.sciencedirect.com/science/article/abs/pii/S0167923609001377) of this dataset the authors report that a modified [Support Vector Machine (SVM)](https://en.wikipedia.org/wiki/Support_vector_machine) model performs better than a deep neural network or decision tree models. 

The last couple of explanations are more plausible, but difficult to prove. For the first one we would need to do more statistical tests to evaluate the contribution of that variable to the prediction. However, it is not hard to think that there is a subjective component (expert preferences, physiological differences) when it comes to evaluating wine quality, predicting human behavior is a complicated task and I wouldn't be surprised that even with a perfect model we would still not be able to get great results because we are lacking information about individual preferences from the experts.

Don't forget to take a look at the notebook for this post [here!](https://github.com/spiralizing/WebsiteNotebooks/blob/main/Python/Wine.ipynb) 

