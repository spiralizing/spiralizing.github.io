+++
title = "Copying the style of an image to another"
hascode = true
date = Date(2022, 10, 14)
rss = "Transfering the style from one image to another"

tags = ["science", "data science"]
+++

Style transfer is one of the many cool applications that [Convolutional Neural Networks (CNNs)](https://en.wikipedia.org/wiki/Convolutional_neural_network) have. The main idea of this method is to use the **kernel features** that a model has learned to transfer the _style_ or **spatial structure** of an image to a different one.
\toc
### Loading imports, model and images
First we are going to load some imports
```python
import seaborn as sns

import scipy.stats as stats

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import torchvision.transforms as T

from torchsummary import summary

import time
import os
import copy
import numpy as np

#importing convolution from scipy
from scipy.signal import convolve2d
#reading image
from imageio import imread
#plotting
import matplotlib.pyplot as plt

```
we also will need to use GPUs since training on the CPU can take way longer, we set our device as `cuda:0`.

```python
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
```
For this particular case we are going to load the [VGG-19](https://iq.opengenus.org/vgg19-architecture/), a well known CNN architecture that has already been trained for image classification. This CNN architecture features 19 layers, 16 convolutional and 3 linear (or fully connected).

```python
vggnet = torchvision.models.vgg19(pretrained=True)

```
Because we are not interested in training the network but only in passing images through it, we need to _lock_ or _freeze_ all its parameters:
```python
#freezing all parameters
for p in vggnet.parameters():
    p.requires_grad = False

#switching to evaluation mode
vggnet.eval()
#moving the model to the GPU
vggnet.to(device)
```
```plaintext
VGG(
  (features): Sequential(
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU(inplace=True)
    (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU(inplace=True)
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (6): ReLU(inplace=True)
    (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (8): ReLU(inplace=True)
    (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): ReLU(inplace=True)
    (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (13): ReLU(inplace=True)
    (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (15): ReLU(inplace=True)
    (16): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (17): ReLU(inplace=True)
    (18): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (19): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (20): ReLU(inplace=True)
    (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (22): ReLU(inplace=True)
    (23): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (24): ReLU(inplace=True)
    (25): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (26): ReLU(inplace=True)
    (27): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (29): ReLU(inplace=True)
    (30): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (31): ReLU(inplace=True)
    (32): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (33): ReLU(inplace=True)
    (34): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (35): ReLU(inplace=True)
    (36): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(7, 7))
  (classifier): Sequential(
    (0): Linear(in_features=25088, out_features=4096, bias=True)
    (1): ReLU(inplace=True)
    (2): Dropout(p=0.5, inplace=False)
    (3): Linear(in_features=4096, out_features=4096, bias=True)
    (4): ReLU(inplace=True)
    (5): Dropout(p=0.5, inplace=False)
    (6): Linear(in_features=4096, out_features=1000, bias=True)
  )
)
```
Now that our model is loaded on the GPU, we are going to leave it there for a bit and load the images we are going to use. 

The first image is going to be the picture I have on my home here, and the image we are going to extract and copy its style is and art work from [Allyson Grey](https://www.allysongrey.com/) (Alex Grey's daughter)
```python
#importing images 
img_content = imread('https://raw.githubusercontent.com/spiralizing/CVResume/main/Resume/Mypic.jpeg')
img_style = imread('http://oregoneclipse2017.com/wp-content/uploads/2017/08/allyson-grey.jpg')
```
we also need to initialize the final image, since we are going to generate a new image by copying features from the two images we loaded, we can simply generate an image with random numbers that lie within the range (0,255).
```python
#initialize the target image with random numbers

img_target = np.random.randint(low=0, high=255, size= img_content.shape, dtype=np.uint8)
#checking sizes

print(img_content.shape)
print(img_style.shape)
print(img_target.shape)
```
```plaintext
(431, 431, 3)
(1600, 1603, 3)
(431, 431, 3)
```
Now we need to make sure we have our images in the right format for pytorch, for this we are going to create a transformation with a normalization, resizing and conversion to tensors
```python
#re-sizing the images so it takes less time to train
Trans = T.Compose(
    [T.ToTensor(), 
    T.Resize(256), 
    #normalization values were extracted from the vgg19 data
    T.Normalize([0.485, 0.456, 0.406], [0.229,0.224,0.225])]
)

#unsqueeze the images to make them a 4D tensor
img_content = Trans( img_content ).unsqueeze(0).to(device)
img_style = Trans( img_style ).unsqueeze(0).to(device)
img_target = Trans( img_target ).unsqueeze(0).to(device)

#check shapes [n_batch, channels, px_y, px_x]
print(img_content.shape)
print(img_style.shape)
print(img_target.shape)
```
```plaintext
torch.Size([1, 3, 256, 256])
torch.Size([1, 3, 256, 256])
torch.Size([1, 3, 256, 256])
```
Now that our images are in the right format we can visualize them before starting with the process of copying the style
```python
fig, ax = plt.subplots(1,3, figsize=(18,6))

titles = ['Content pic', 'New pic', 'Style pic']

for i, pic in enumerate([img_content, img_target, img_style]):
    img = pic.cpu().squeeze().numpy().transpose((1,2,0)) #transform for display
    img = (img - np.min(img)) / (np.max(img)-np.min(img)) #undo normalization
    ax[i].imshow(img)
    ax[i].set_title(titles[i])
```
~~~
<div class="container">

    <img class="center" src="/assets/transfer_img1.svg" width="500" height="350">

</div>
~~~
the _new pic_ is the image that we are going to modify to make it look like the content and style images. Before doing that we need to define a couple of functions that will help us to extract the feature activation maps and compute the gram matrix (or covariance matrix). 
### Feature activation maps and gram matrices
The feature activation maps are obtained by passing the image through the kernel(s) of every layer in the model, in this case we have 16 convolutional layers so we will have 16 images

```python
def get_feat_actmaps(img, net):
    feature_maps = []
    feature_names = []

    convL_ix = 0 #counter init

    #loop over the layers in the features block
    for lay_num in range(len(net.features)):
        #process the image through this layer
        img = net.features[lay_num](img)
        #store the results that come from the convolutional layers
        if 'Conv2d' in str(net.features[lay_num]):
            feature_maps.append( img )
            feature_names.append('ConvLayer_' + str(convL_ix))
            convL_ix += 1
        
    return feature_maps, feature_names

def get_gramMat(M):
    #reshaping to 2D
    _,chans,height,width = M.shape
    M = M.reshape(chans, height*width) 

    #compute covariance matrix
    gram = torch.mm(M, M.t()) / (chans*height*width)

    return gram
```
we then apply our feature activation map function to the content image using the VGG-19 model

```python
content_fm, content_fn = get_feat_actmaps(img_content, vggnet)
```
now we plot each of the image that results from the convolution between the kernel(s) in the layers and the original image, with their respective gram matrices
```python
fig, axs = plt.subplots(2,7, figsize=(18,6))

for i in range(7):
    img = np.mean( content_fm[i].cpu().squeeze().numpy(), axis=0)
    img = (img - np.min(img))/(np.max(img) - np.min(img))

    axs[0,i].imshow(img, cmap='gray')
    axs[0,i].set_title('Content'+ str(content_fn[i]))

    #the gram matrix:
    img = get_gramMat(content_fm[i]).cpu().numpy()
    img = (img - np.min(img))/(np.max(img)-np.min(img))

    axs[1,i].imshow(img, cmap='gray',vmax=0.1)
    axs[1,i].set_title('GramMat '+ str(content_fn[i]))

plt.tight_layout()
plt.show()
```
~~~
<div class="container">

    <img class="center" src="/assets/transfer_gram1.svg" width="500" height="350">

</div>
~~~
it is worth reminding that these results come from a pre trained model that already learned the features (kernels) from a set of images, we are only exploring how those features are extracted from our image.

As expected, the gram matrix is symmetric and includes nontrivial spatial structure from the feature activation map since it is a type non normalized [correlation matrix](https://en.wikipedia.org/wiki/Correlation#Correlation_matrices), so it includes statistical dependencies between pixels.

We now compute the same for the _style_ image
```python
style_fm, style_fn = get_feat_actmaps(img_style, vggnet)

fig, axs = plt.subplots(2, 7, figsize=(18, 6))

for i in range(7):
    img = np.mean(style_fm[i].cpu().squeeze().numpy(), axis=0)
    img = (img - np.min(img))/(np.max(img) - np.min(img))

    axs[0, i].imshow(img, cmap='gray')
    axs[0, i].set_title('style ' + str(style_fn[i]))

    #the gram matrix:
    img = get_gramMat(style_fm[i]).cpu().numpy()
    img = (img - np.min(img))/(np.max(img)-np.min(img))

    axs[1, i].imshow(img, cmap='gray', vmax=0.1)
    axs[1, i].set_title('GramMat ' + str(style_fn[i]))

plt.tight_layout()
plt.show()
```
~~~
<div class="container">

    <img class="center" src="/assets/transfer_gram2.svg" width="500" height="350">

</div>
~~~

The next step is a bit more trial and error, we need to choose what information we want to copy from the processed images, and decide _how much_ we want to copy from each layer 
```python
#2 layers from content
layers_content = ['ConvLayer_1', 'ConvLayer_2']
#5 layers for style
layers_style = ['ConvLayer_1','ConvLayer_2','ConvLayer_3','ConvLayer_4','ConvLayer_5']
#how much weight to give to each style layer
weights_style = [1, 0.5, 0.5, 0.2 ,0.1] 
```
for the case of the content picture we are going to use only layers 1 and 2, and layers 1-5 for the style picture with their respective weights `weights_style`. This is arbitrary and it depends on what information you want to copy, so I recommend take a look at the images and decide what shapes, lines etc you would like to preserve.

### Training (creating) the image
Since our model is already trained, it might be confusing what exactly are we going to train and optimize. To figure this out we just need to remember what is our main goal, which is to _transfer_ the style of one image to another. What we want to modify is our randomly generated image, this means that our loss should be a combination of losses between our _noisy_ image and each of the content and style images.

$$ total\_loss = content\_loss + style\_loss $$

To achieve this we need to set the optimizer acting on the image we are modifying and our loss functions can be a [mean squared error (MSE)](https://en.wikipedia.org/wiki/Mean_squared_error) between the _noisy_ or _target_ image and each of the other images. 

The **trick** here is to understand that we are trying to make the final image to look like the content image with the **structural properties** of the style image, these structural properties are in the gram matrices for the _style_ image. This means we need to compute the content loss as the MSE between the target and content activation maps, and the style loss as the MSE between the gram matrices of the target and style activation maps. Each loss computation as a linear combination with [1,1] weights for content loss and `weights_style` weights for style loss.

```python
#the image that we are going to train
target = img_target.clone()
target.requires_grad = True
target = target.to(device)
#scale up the loss function for the style, this adds more 'importance' to the style
style_scale = 1e5 

n_epochs = 2500
#optimizing the target image
optimizer = torch.optim.RMSprop([target], lr=0.005)
```

```python
for e_i in range(n_epochs):
    target_fm, target_fn = get_feat_actmaps(target, vggnet)

    style_loss = 0
    content_loss = 0

    for layer_i in range(len(target_fn)):
        #using only the layers specified previously

        #content loss
        if target_fn[layer_i] in layers_content:
            content_loss += torch.mean(( target_fm[layer_i] - content_fm[layer_i])**2)
        
        #style loss
        if target_fn[layer_i] in layers_style:
            #computing gram Matrices
            Gtarget = get_gramMat(target_fm[layer_i])
            Gstyle = get_gramMat(style_fm[layer_i])

            #compute loss with weights
            style_loss += torch.mean( (Gtarget - Gstyle)**2 ) * weights_style[layers_style.index(target_fn[layer_i])]

    #computing combined loss (re-scaled style loss + content loss)
    comb_loss = style_scale*style_loss + content_loss

    #backprop
    optimizer.zero_grad()
    comb_loss.backward()
    optimizer.step()
```
### Final result

Now that we _trained_ our target image, we can finally see the resulting image

```python
fig, ax = plt.subplots(1, 3, figsize=(18, 11))

pic = img_content.cpu().squeeze().numpy().transpose((1, 2, 0))
pic = (pic-np.min(pic)) / (np.max(pic)-np.min(pic))
ax[0].imshow(pic)
ax[0].set_title('Content picture', fontweight='bold')
ax[0].set_xticks([])
ax[0].set_yticks([])

pic = torch.sigmoid(target).cpu().detach(
).squeeze().numpy().transpose((1, 2, 0))
ax[1].imshow(pic)
ax[1].set_title('New picture', fontweight='bold')
ax[1].set_xticks([])
ax[1].set_yticks([])

pic = img_style.cpu().squeeze().numpy().transpose((1, 2, 0))
pic = (pic-np.min(pic)) / (np.max(pic)-np.min(pic))
ax[2].imshow(pic)
ax[2].set_title('Style picture', fontweight='bold')
ax[2].set_xticks([])
ax[2].set_yticks([])

plt.show()
```
~~~
<div class="container">

    <img class="center" src="/assets/transfer_final.svg" width="500" height="350">

</div>
~~~

which I kind of like it better than the original, I will consider using it as a profile picture instead.

Don't forget to take a look at the [notebook](https://github.com/spiralizing/WebsiteNotebooks/blob/main/Python/StyleTransfer.ipynb) for this post.