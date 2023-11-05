## Prepare the workspace


```python
# Before you proceed, update the PATH
import os
os.environ['PATH'] = f"{os.environ['PATH']}:/root/.local/bin"
os.environ['PATH'] = f"{os.environ['PATH']}:/opt/conda/lib/python3.6/site-packages"
# Restart the Kernel at this point. 
```


```python
# Do not execute the commands below unless you have restart the Kernel after updating the PATH. 
!python -m pip install torch==2.1.0
```

    Defaulting to user installation because normal site-packages is not writeable
    Collecting torch==2.1.0
      Downloading torch-2.1.0-cp310-cp310-manylinux1_x86_64.whl (670.2 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m670.2/670.2 MB[0m [31m1.2 MB/s[0m eta [36m0:00:00[0m00:01[0m00:01[0m
    [?25hCollecting nvidia-cudnn-cu12==8.9.2.26
      Downloading nvidia_cudnn_cu12-8.9.2.26-py3-none-manylinux1_x86_64.whl (731.7 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m731.7/731.7 MB[0m [31m1.2 MB/s[0m eta [36m0:00:00[0m00:01[0m00:01[0m
    [?25hRequirement already satisfied: networkx in /opt/conda/lib/python3.10/site-packages (from torch==2.1.0) (3.1)
    Collecting nvidia-nvtx-cu12==12.1.105
      Downloading nvidia_nvtx_cu12-12.1.105-py3-none-manylinux1_x86_64.whl (99 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m99.1/99.1 kB[0m [31m12.5 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting nvidia-cufft-cu12==11.0.2.54
      Downloading nvidia_cufft_cu12-11.0.2.54-py3-none-manylinux1_x86_64.whl (121.6 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m121.6/121.6 MB[0m [31m8.2 MB/s[0m eta [36m0:00:00[0m00:01[0m00:01[0m
    [?25hRequirement already satisfied: sympy in /opt/conda/lib/python3.10/site-packages (from torch==2.1.0) (1.12)
    Collecting nvidia-cuda-runtime-cu12==12.1.105
      Downloading nvidia_cuda_runtime_cu12-12.1.105-py3-none-manylinux1_x86_64.whl (823 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m823.6/823.6 kB[0m [31m32.9 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting nvidia-cuda-cupti-cu12==12.1.105
      Downloading nvidia_cuda_cupti_cu12-12.1.105-py3-none-manylinux1_x86_64.whl (14.1 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m14.1/14.1 MB[0m [31m40.0 MB/s[0m eta [36m0:00:00[0m00:01[0m00:01[0m
    [?25hCollecting nvidia-cuda-nvrtc-cu12==12.1.105
      Downloading nvidia_cuda_nvrtc_cu12-12.1.105-py3-none-manylinux1_x86_64.whl (23.7 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m23.7/23.7 MB[0m [31m34.5 MB/s[0m eta [36m0:00:00[0m00:01[0m00:01[0m
    [?25hRequirement already satisfied: filelock in /opt/conda/lib/python3.10/site-packages (from torch==2.1.0) (3.9.0)
    Collecting nvidia-cublas-cu12==12.1.3.1
      Downloading nvidia_cublas_cu12-12.1.3.1-py3-none-manylinux1_x86_64.whl (410.6 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m410.6/410.6 MB[0m [31m2.2 MB/s[0m eta [36m0:00:00[0m00:01[0m00:01[0m
    [?25hCollecting nvidia-nccl-cu12==2.18.1
      Downloading nvidia_nccl_cu12-2.18.1-py3-none-manylinux1_x86_64.whl (209.8 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m209.8/209.8 MB[0m [31m4.4 MB/s[0m eta [36m0:00:00[0m00:01[0m00:01[0m
    [?25hCollecting nvidia-cusolver-cu12==11.4.5.107
      Downloading nvidia_cusolver_cu12-11.4.5.107-py3-none-manylinux1_x86_64.whl (124.2 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m124.2/124.2 MB[0m [31m7.8 MB/s[0m eta [36m0:00:00[0m00:01[0m00:01[0m
    [?25hRequirement already satisfied: jinja2 in /opt/conda/lib/python3.10/site-packages (from torch==2.1.0) (3.1.2)
    Requirement already satisfied: typing-extensions in /opt/conda/lib/python3.10/site-packages (from torch==2.1.0) (4.5.0)
    Collecting fsspec
      Downloading fsspec-2023.10.0-py3-none-any.whl (166 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m166.4/166.4 kB[0m [31m17.8 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting nvidia-curand-cu12==10.3.2.106
      Downloading nvidia_curand_cu12-10.3.2.106-py3-none-manylinux1_x86_64.whl (56.5 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m56.5/56.5 MB[0m [31m15.7 MB/s[0m eta [36m0:00:00[0m00:01[0m00:01[0m
    [?25hCollecting nvidia-cusparse-cu12==12.1.0.106
      Downloading nvidia_cusparse_cu12-12.1.0.106-py3-none-manylinux1_x86_64.whl (196.0 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m196.0/196.0 MB[0m [31m5.1 MB/s[0m eta [36m0:00:00[0m00:01[0m00:01[0m
    [?25hCollecting triton==2.1.0
      Downloading triton-2.1.0-0-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (89.2 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m89.2/89.2 MB[0m [31m10.8 MB/s[0m eta [36m0:00:00[0m00:01[0m00:01[0m
    [?25hCollecting nvidia-nvjitlink-cu12
      Downloading nvidia_nvjitlink_cu12-12.3.52-py3-none-manylinux1_x86_64.whl (20.5 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m20.5/20.5 MB[0m [31m15.4 MB/s[0m eta [36m0:00:00[0m00:01[0m00:01[0m
    [?25hRequirement already satisfied: MarkupSafe>=2.0 in /opt/conda/lib/python3.10/site-packages (from jinja2->torch==2.1.0) (2.1.1)
    Requirement already satisfied: mpmath>=0.19 in /opt/conda/lib/python3.10/site-packages (from sympy->torch==2.1.0) (1.3.0)
    Installing collected packages: triton, nvidia-nvtx-cu12, nvidia-nvjitlink-cu12, nvidia-nccl-cu12, nvidia-curand-cu12, nvidia-cufft-cu12, nvidia-cuda-runtime-cu12, nvidia-cuda-nvrtc-cu12, nvidia-cuda-cupti-cu12, nvidia-cublas-cu12, fsspec, nvidia-cusparse-cu12, nvidia-cudnn-cu12, nvidia-cusolver-cu12, torch
    [33m  WARNING: The scripts convert-caffe2-to-onnx, convert-onnx-to-caffe2 and torchrun are installed in '/home/student/.local/bin' which is not on PATH.
      Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.[0m[33m
    [0mSuccessfully installed fsspec-2023.10.0 nvidia-cublas-cu12-12.1.3.1 nvidia-cuda-cupti-cu12-12.1.105 nvidia-cuda-nvrtc-cu12-12.1.105 nvidia-cuda-runtime-cu12-12.1.105 nvidia-cudnn-cu12-8.9.2.26 nvidia-cufft-cu12-11.0.2.54 nvidia-curand-cu12-10.3.2.106 nvidia-cusolver-cu12-11.4.5.107 nvidia-cusparse-cu12-12.1.0.106 nvidia-nccl-cu12-2.18.1 nvidia-nvjitlink-cu12-12.3.52 nvidia-nvtx-cu12-12.1.105 torch-2.1.0 triton-2.1.0



```python
# Check torch version and CUDA status if GPU is enabled.
!pip install --upgrade pytorch

import torch

print(torch.__version__)
print(torch.cuda.is_available()) # Should return True when GPU is enabled. 
```

    Defaulting to user installation because normal site-packages is not writeable
    Collecting pytorch
      Using cached pytorch-1.0.2.tar.gz (689 bytes)
      Preparing metadata (setup.py) ... [?25ldone
    [?25hBuilding wheels for collected packages: pytorch
      Building wheel for pytorch (setup.py) ... [?25lerror
      [1;31merror[0m: [1msubprocess-exited-with-error[0m
      
      [31m×[0m [32mpython setup.py bdist_wheel[0m did not run successfully.
      [31m│[0m exit code: [1;36m1[0m
      [31m╰─>[0m [31m[6 lines of output][0m
      [31m   [0m Traceback (most recent call last):
      [31m   [0m   File "<string>", line 2, in <module>
      [31m   [0m   File "<pip-setuptools-caller>", line 34, in <module>
      [31m   [0m   File "/tmp/pip-install-uk_7codp/pytorch_a0a80374955c4c46ac0ede9625ded76d/setup.py", line 15, in <module>
      [31m   [0m     raise Exception(message)
      [31m   [0m Exception: You tried to install "pytorch". The package named for PyTorch is "torch"
      [31m   [0m [31m[end of output][0m
      
      [1;35mnote[0m: This error originates from a subprocess, and is likely not a problem with pip.
    [31m  ERROR: Failed building wheel for pytorch[0m[31m
    [0m[?25h  Running setup.py clean for pytorch
    Failed to build pytorch
    Installing collected packages: pytorch
      Running setup.py install for pytorch ... [?25lerror
      [1;31merror[0m: [1msubprocess-exited-with-error[0m
      
      [31m×[0m [32mRunning setup.py install for pytorch[0m did not run successfully.
      [31m│[0m exit code: [1;36m1[0m
      [31m╰─>[0m [31m[6 lines of output][0m
      [31m   [0m Traceback (most recent call last):
      [31m   [0m   File "<string>", line 2, in <module>
      [31m   [0m   File "<pip-setuptools-caller>", line 34, in <module>
      [31m   [0m   File "/tmp/pip-install-uk_7codp/pytorch_a0a80374955c4c46ac0ede9625ded76d/setup.py", line 11, in <module>
      [31m   [0m     raise Exception(message)
      [31m   [0m Exception: You tried to install "pytorch". The package named for PyTorch is "torch"
      [31m   [0m [31m[end of output][0m
      
      [1;35mnote[0m: This error originates from a subprocess, and is likely not a problem with pip.
    [1;31merror[0m: [1mlegacy-install-failure[0m
    
    [31m×[0m Encountered error while trying to install package.
    [31m╰─>[0m pytorch
    
    [1;35mnote[0m: This is an issue with the package mentioned above, not pip.
    [1;36mhint[0m: See above for output from the failure.
    [?25h2.0.1
    True


# Developing an AI application

Going forward, AI algorithms will be incorporated into more and more everyday applications. For example, you might want to include an image classifier in a smart phone app. To do this, you'd use a deep learning model trained on hundreds of thousands of images as part of the overall application architecture. A large part of software development in the future will be using these types of models as common parts of applications. 

In this project, you'll train an image classifier to recognize different species of flowers. You can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at. In practice you'd train this classifier, then export it for use in your application. We'll be using [this dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) of 102 flower categories, you can see a few examples below. 

<img src='assets/Flowers.png' width=500px>

The project is broken down into multiple steps:

* Load and preprocess the image dataset
* Train the image classifier on your dataset
* Use the trained classifier to predict image content

We'll lead you through each part which you'll implement in Python.

When you've completed this project, you'll have an application that can be trained on any set of labeled images. Here your network will be learning about flowers and end up as a command line application. But, what you do with your new skills depends on your imagination and effort in building a dataset. For example, imagine an app where you take a picture of a car, it tells you what the make and model is, then looks up information about it. Go build your own dataset and make something new.

First up is importing the packages you'll need. It's good practice to keep all the imports at the beginning of your code. As you work through this notebook and find you need to import a package, make sure to add the import up here.


```python
# Imports here

import time
import json
import copy
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image
from collections import OrderedDict
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import os


```

## Load the data

Here you'll use `torchvision` to load the data ([documentation](http://pytorch.org/docs/0.3.0/torchvision/index.html)). The data should be included alongside this notebook, otherwise you can [download it here](https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz). 

If you do not find the `flowers/` dataset in the current directory, **/workspace/home/aipnd-project/**, you can download it using the following commands. 

```bash
!wget 'https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz'
!unlink flowers
!mkdir flowers && tar -xzf flower_data.tar.gz -C flowers
```


## Data Description
The dataset is split into three parts, training, validation, and testing. For the training, you'll want to apply transformations such as random scaling, cropping, and flipping. This will help the network generalize leading to better performance. You'll also need to make sure the input data is resized to 224x224 pixels as required by the pre-trained networks.

The validation and testing sets are used to measure the model's performance on data it hasn't seen yet. For this you don't want any scaling or rotation transformations, but you'll need to resize then crop the images to the appropriate size.

The pre-trained networks you'll use were trained on the ImageNet dataset where each color channel was normalized separately. For all three sets you'll need to normalize the means and standard deviations of the images to what the network expects. For the means, it's `[0.485, 0.456, 0.406]` and for the standard deviations `[0.229, 0.224, 0.225]`, calculated from the ImageNet images.  These values will shift each color channel to be centered at 0 and range from -1 to 1.
 


```python
data_dir = 'flowers'

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}
```


```python
# Create datasets for training, validation, and testing
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'valid', 'test']}

# Create data loaders for each dataset
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=16, shuffle=True, num_workers=0) for x in ['train', 'valid', 'test']}

# Get the size of each dataset
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid', 'test']}
class_names = image_datasets['train'].classes

```

### Label mapping

You'll also need to load in a mapping from category label to category name. You can find this in the file `cat_to_name.json`. It's a JSON object which you can read in with the [`json` module](https://docs.python.org/2/library/json.html). This will give you a dictionary mapping the integer encoded categories to the actual names of the flowers.


```python
import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
```

# Building and training the classifier

Now that the data is ready, it's time to build and train the classifier. As usual, you should use one of the pretrained models from `torchvision.models` to get the image features. Build and train a new feed-forward classifier using those features.

We're going to leave this part up to you. Refer to [the rubric](https://review.udacity.com/#!/rubrics/1663/view) for guidance on successfully completing this section. Things you'll need to do:

* Load a [pre-trained network](http://pytorch.org/docs/master/torchvision/models.html) (If you need a starting point, the VGG networks work great and are straightforward to use)
* Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
* Train the classifier layers using backpropagation using the pre-trained network to get the features
* Track the loss and accuracy on the validation set to determine the best hyperparameters

We've left a cell open for you below, but use as many as you need. Our advice is to break the problem up into smaller parts you can run separately. Check that each part is doing what you expect, then move on to the next. You'll likely find that as you work through each part, you'll need to go back and modify your previous code. This is totally normal!

When training make sure you're updating only the weights of the feed-forward network. You should be able to get the validation accuracy above 70% if you build everything right. Make sure to try different hyperparameters (learning rate, units in the classifier, epochs, etc) to find the best model. Save those hyperparameters to use as default values in the next part of the project.

One last important tip if you're using the workspace to run your code: To avoid having your workspace disconnect during the long-running tasks in this notebook, please read in the earlier page in this lesson called Intro to
GPU Workspaces about Keeping Your Session Active. You'll want to include code from the workspace_utils.py module.

## Note for Workspace users: 
If your network is over 1 GB when saved as a checkpoint, there might be issues with saving backups in your workspace. Typically this happens with wide dense layers after the convolutional layers. If your saved checkpoint is larger than 1 GB (you can open a terminal and check with `ls -lh`), you should reduce the size of your hidden layers and train again.


```python
# TODO: Build and train your network

model = models.vgg19(weights=True)
model

classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, 4096)),
                          ('relu', nn.ReLU()),
                          ('dropout', nn.Dropout(p = 0.5)),
                          ('fc2', nn.Linear(4096, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))

for param in model.parameters():
    param.requires_grad = False
    
model.classifier = classifier
#model
```


```python
def train_model(model, criterion, optimizer, sched,   
                                      num_epochs=25, device="cuda"):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        model.to(device)
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                sched.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

criterion = nn.NLLLoss()

optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

sched = lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)

eps=4
```


```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device('mps' if torch.backends.mps.is_available()  else 'cpu')
print("device={}".format(device))
model.to(device)
```

    device=cuda



    ---------------------------------------------------------------------------

    OutOfMemoryError                          Traceback (most recent call last)

    Cell In[14], line 4
          2 #device = torch.device('mps' if torch.backends.mps.is_available()  else 'cpu')
          3 print("device={}".format(device))
    ----> 4 model.to(device)


    File /opt/conda/lib/python3.10/site-packages/torch/nn/modules/module.py:1145, in Module.to(self, *args, **kwargs)
       1141         return t.to(device, dtype if t.is_floating_point() or t.is_complex() else None,
       1142                     non_blocking, memory_format=convert_to_format)
       1143     return t.to(device, dtype if t.is_floating_point() or t.is_complex() else None, non_blocking)
    -> 1145 return self._apply(convert)


    File /opt/conda/lib/python3.10/site-packages/torch/nn/modules/module.py:797, in Module._apply(self, fn)
        795 def _apply(self, fn):
        796     for module in self.children():
    --> 797         module._apply(fn)
        799     def compute_should_use_set_data(tensor, tensor_applied):
        800         if torch._has_compatible_shallow_copy_type(tensor, tensor_applied):
        801             # If the new tensor has compatible tensor type as the existing tensor,
        802             # the current behavior is to change the tensor in-place using `.data =`,
       (...)
        807             # global flag to let the user control whether they want the future
        808             # behavior of overwriting the existing tensor or not.


    File /opt/conda/lib/python3.10/site-packages/torch/nn/modules/module.py:797, in Module._apply(self, fn)
        795 def _apply(self, fn):
        796     for module in self.children():
    --> 797         module._apply(fn)
        799     def compute_should_use_set_data(tensor, tensor_applied):
        800         if torch._has_compatible_shallow_copy_type(tensor, tensor_applied):
        801             # If the new tensor has compatible tensor type as the existing tensor,
        802             # the current behavior is to change the tensor in-place using `.data =`,
       (...)
        807             # global flag to let the user control whether they want the future
        808             # behavior of overwriting the existing tensor or not.


    File /opt/conda/lib/python3.10/site-packages/torch/nn/modules/module.py:820, in Module._apply(self, fn)
        816 # Tensors stored in modules are graph leaves, and we don't want to
        817 # track autograd history of `param_applied`, so we have to use
        818 # `with torch.no_grad():`
        819 with torch.no_grad():
    --> 820     param_applied = fn(param)
        821 should_use_set_data = compute_should_use_set_data(param, param_applied)
        822 if should_use_set_data:


    File /opt/conda/lib/python3.10/site-packages/torch/nn/modules/module.py:1143, in Module.to.<locals>.convert(t)
       1140 if convert_to_format is not None and t.dim() in (4, 5):
       1141     return t.to(device, dtype if t.is_floating_point() or t.is_complex() else None,
       1142                 non_blocking, memory_format=convert_to_format)
    -> 1143 return t.to(device, dtype if t.is_floating_point() or t.is_complex() else None, non_blocking)


    OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 MiB (GPU 0; 11.17 GiB total capacity; 598.80 MiB already allocated; 7.25 MiB free; 616.00 MiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF



```python
model_ft = train_model(model, criterion, optimizer, sched, eps, "cuda")
```


    ---------------------------------------------------------------------------

    OutOfMemoryError                          Traceback (most recent call last)

    Cell In[11], line 1
    ----> 1 model_ft = train_model(model, criterion, optimizer, sched, eps, "cuda")


    Cell In[9], line 5, in train_model(model, criterion, optimizer, sched, num_epochs, device)
          1 def train_model(model, criterion, optimizer, sched,   
          2                                       num_epochs=25, device="cuda"):
          3     since = time.time()
    ----> 5     best_model_wts = copy.deepcopy(model.state_dict())
          6     best_acc = 0.0
          8     for epoch in range(num_epochs):


    File /opt/conda/lib/python3.10/copy.py:172, in deepcopy(x, memo, _nil)
        170                 y = x
        171             else:
    --> 172                 y = _reconstruct(x, memo, *rv)
        174 # If is its own copy, don't memoize.
        175 if y is not x:


    File /opt/conda/lib/python3.10/copy.py:297, in _reconstruct(x, memo, func, args, state, listiter, dictiter, deepcopy)
        295     for key, value in dictiter:
        296         key = deepcopy(key, memo)
    --> 297         value = deepcopy(value, memo)
        298         y[key] = value
        299 else:


    File /opt/conda/lib/python3.10/copy.py:153, in deepcopy(x, memo, _nil)
        151 copier = getattr(x, "__deepcopy__", None)
        152 if copier is not None:
    --> 153     y = copier(memo)
        154 else:
        155     reductor = dispatch_table.get(cls)


    File /opt/conda/lib/python3.10/site-packages/torch/_tensor.py:118, in Tensor.__deepcopy__(self, memo)
        109         raise RuntimeError(
        110             "The default implementation of __deepcopy__() for wrapper subclasses "
        111             "only works for subclass types that implement clone() and for which "
       (...)
        115             "different type."
        116         )
        117 else:
    --> 118     new_storage = self._typed_storage()._deepcopy(memo)
        119     if self.is_quantized:
        120         # quantizer_params can be different type based on torch attribute
        121         quantizer_params: Union[
        122             Tuple[torch.qscheme, float, int],
        123             Tuple[torch.qscheme, Tensor, Tensor, int],
        124         ]


    File /opt/conda/lib/python3.10/site-packages/torch/storage.py:684, in TypedStorage._deepcopy(self, memo)
        683 def _deepcopy(self, memo):
    --> 684     return self._new_wrapped_storage(copy.deepcopy(self._untyped_storage, memo))


    File /opt/conda/lib/python3.10/copy.py:153, in deepcopy(x, memo, _nil)
        151 copier = getattr(x, "__deepcopy__", None)
        152 if copier is not None:
    --> 153     y = copier(memo)
        154 else:
        155     reductor = dispatch_table.get(cls)


    File /opt/conda/lib/python3.10/site-packages/torch/storage.py:98, in _StorageBase.__deepcopy__(self, memo)
         96 if self._cdata in memo:
         97     return memo[self._cdata]
    ---> 98 new_storage = self.clone()
         99 memo[self._cdata] = new_storage
        100 return new_storage


    File /opt/conda/lib/python3.10/site-packages/torch/storage.py:112, in _StorageBase.clone(self)
        110 def clone(self):
        111     """Returns a copy of this storage"""
    --> 112     return type(self)(self.nbytes(), device=self.device).copy_(self)


    OutOfMemoryError: CUDA out of memory. Tried to allocate 392.00 MiB (GPU 0; 11.17 GiB total capacity; 548.54 MiB already allocated; 47.25 MiB free; 576.00 MiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF


## Testing your network

It's good practice to test your trained network on test data, images the network has never seen either in training or validation. This will give you a good estimate for the model's performance on completely new images. Run the test images through the network and measure the accuracy, the same way you did validation. You should be able to reach around 70% accuracy on the test set if the model has been trained well.


```python
# TODO: Do validation on the test set

def calc_accuracy(model, data, cuda=False):
    model.eval()
    model.to(device='cuda')    
    
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(dataloaders[data]):
            if cuda:
                inputs, labels = inputs.cuda(), labels.cuda()
            # obtain the outputs from the model
            outputs = model.forward(inputs)
            # max provides the (maximum probability, max value)
            _, predicted = outputs.max(dim=1)
            # check the 
            if idx == 0:
                print(predicted) #the predicted class
                print(torch.exp(_)) # the predicted probability
            equals = predicted == labels.data
            if idx == 0:
                print(equals)
            print(equals.float().mean())
            
calc_accuracy(model, 'test', True)
```

## Save the checkpoint

Now that your network is trained, save the model so you can load it later for making predictions. You probably want to save other things such as the mapping of classes to indices which you get from one of the image datasets: `image_datasets['train'].class_to_idx`. You can attach this to the model as an attribute which makes inference easier later on.

```model.class_to_idx = image_datasets['train'].class_to_idx```

Remember that you'll want to completely rebuild the model later so you can use it for inference. Make sure to include any information you need in the checkpoint. If you want to load the model and keep training, you'll want to save the number of epochs as well as the optimizer state, `optimizer.state_dict`. You'll likely want to use this trained model in the next part of the project, so best to save it now.


```python
# TODO: Save the checkpoint 

model.class_to_idx = image_datasets['train'].class_to_idx
model.cpu()
torch.save({'arch': 'vgg19',
            'state_dict': model.state_dict(), 
            'class_to_idx': model.class_to_idx},
#            'Num_epochs': num_epochs(),
#            'optimizer': optimizer.state_dict,
            'classifier.pth')
```

## Loading the checkpoint

At this point it's good to write a function that can load a checkpoint and rebuild the model. That way you can come back to this project and keep working on it without having to retrain the network.


```python
# TODO: Write a function that loads a checkpoint and rebuilds the model

def load_model(checkpoint_path):
    chpt = torch.load(checkpoint_path)
    
    if chpt['arch'] == 'vgg19':
        model = models.vgg19(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
    elif:
        print("Sorry - base architecture unrecognized, please match architectures.")
        break 
    
    model.class_to_idx = chpt['class_to_idx']
    
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, 4096)),
                          ('relu', nn.ReLU()),
                          ('dropout', nn.Dropout(p = 0.5)),
                          ('fc2', nn.Linear(4096, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
    # Put the classifier on the pretrained network
    model.classifier = classifier
    
    model.load_state_dict(chpt['state_dict'])
    
    return model



#Testing that load has worked correctly - should achieve similiar results as those prior to load. 

model = load_model('classifier.pth')
calc_accuracy(model, 'test', True)
```

# Inference for classification

Now you'll write a function to use a trained network for inference. That is, you'll pass an image into the network and predict the class of the flower in the image. Write a function called `predict` that takes an image and a model, then returns the top $K$ most likely classes along with the probabilities. It should look like 

```python
probs, classes = predict(image_path, model)
print(probs)
print(classes)
> [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
> ['70', '3', '45', '62', '55']
```

First you'll need to handle processing the input image such that it can be used in your network. 

## Image Preprocessing

You'll want to use `PIL` to load the image ([documentation](https://pillow.readthedocs.io/en/latest/reference/Image.html)). It's best to write a function that preprocesses the image so it can be used as input for the model. This function should process the images in the same manner used for training. 

First, resize the images where the shortest side is 256 pixels, keeping the aspect ratio. This can be done with the [`thumbnail`](http://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.thumbnail) or [`resize`](http://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.thumbnail) methods. Then you'll need to crop out the center 224x224 portion of the image.

Color channels of images are typically encoded as integers 0-255, but the model expected floats 0-1. You'll need to convert the values. It's easiest with a Numpy array, which you can get from a PIL image like so `np_image = np.array(pil_image)`.

As before, the network expects the images to be normalized in a specific way. For the means, it's `[0.485, 0.456, 0.406]` and for the standard deviations `[0.229, 0.224, 0.225]`. You'll want to subtract the means from each color channel, then divide by the standard deviation. 

And finally, PyTorch expects the color channel to be the first dimension but it's the third dimension in the PIL image and Numpy array. You can reorder dimensions using [`ndarray.transpose`](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.ndarray.transpose.html). The color channel needs to be first and retain the order of the other two dimensions.


```python
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    
    from PIL import Image
    image_path = 'test/28/image_05230.jpg'
    img = Image.open(image_path)
    
    #Resizing
    if img.size[0] > img.size[1]:
        img.thumbnail(200000, 256) 
    else:
        img.thumbnail((256, 200000))
        
    #Cropping image    
    left_margin = (img.width-224)/2
    bottom_margin = (img.height-224)/2
    right_margin = left_margin + 224
    top_margin = bottom_margin + 224
    
    img = img.crop((left_margin, bottom_margin, right_margin,    
                   top_margin))
    
    #Colour channel as floats
    img = np.array(img)/255
    
    #Normalize the images as expected by network
    mean = np.array([0.485, 0.456, 0.406]) #provided mean
    std = np.array([0.229, 0.224, 0.225]) #provided std
    img = (img - mean)/std
    
    #Reorder the colour channel to be first for PyTorch
    img_final = img.transpose((2, 0, 1))
    
    return img_final
                 
```

To check your work, the function below converts a PyTorch tensor and displays it in the notebook. If your `process_image` function works, running the output through this function should return the original image (except for the cropped out portions).


```python
def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
        
    if title:
        plt.title(title)
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax


image_path = 'test/28/image_05230.jpg'
img = process_image(image_path)
imshow(img)
```

## Class Prediction

Once you can get images in the correct format, it's time to write a function for making predictions with your model. A common practice is to predict the top 5 or so (usually called top-$K$) most probable classes. You'll want to calculate the class probabilities then find the $K$ largest values.

To get the top $K$ largest values in a tensor use [`x.topk(k)`](http://pytorch.org/docs/master/torch.html#torch.topk). This method returns both the highest `k` probabilities and the indices of those probabilities corresponding to the classes. You need to convert from these indices to the actual class labels using `class_to_idx` which hopefully you added to the model or from an `ImageFolder` you used to load the data ([see here](#Save-the-checkpoint)). Make sure to invert the dictionary so you get a mapping from index to class as well.

Again, this method should take a path to an image and a model checkpoint, then return the probabilities and classes.

```python
probs, classes = predict(image_path, model)
print(probs)
print(classes)
> [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
> ['70', '3', '45', '62', '55']
```


```python
def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    
    # Process image
    img = process_image(image_path)
    
    # Make tensor instead of Numpy
    image_tensor = torch.from_numpy(img).type(torch.FloatTensor)
    # Add batch of size 1 to image
    model_input = image_tensor.unsqueeze(0)
    
    # Probs
    probs = torch.exp(model.forward(model_input))
    
    # Top probs
    top_probs, top_labs = probs.topk(top_num)
    top_probs = top_probs.detach().numpy().tolist()[0] 
    top_labs = top_labs.detach().numpy().tolist()[0]
    
    # Convert indices to classes
    idx_to_class = {val: key for key, val in    
                                      model.class_to_idx.items()}
    top_labels = [idx_to_class[lab] for lab in top_labs]
    top_flowers = [label_map[idx_to_class[lab]] for lab in top_labs]
    return top_probs, top_labels, top_flowers

```

## Sanity Checking

Now that you can use a trained model for predictions, check to make sure it makes sense. Even if the testing accuracy is high, it's always good to check that there aren't obvious bugs. Use `matplotlib` to plot the probabilities for the top 5 classes as a bar graph, along with the input image. It should look like this:

<img src='assets/inference_example.png' width=300px>

You can convert from the class integer encoding to actual flower names with the `cat_to_name.json` file (should have been loaded earlier in the notebook). To show a PyTorch tensor as an image, use the `imshow` function defined above.


```python
# TODO: Display an image along with the top 5 classes

def plot_solution(image_path, model):
    # Set up plot
    plt.figure(figsize = (6,10))
    ax = plt.subplot(2,1,1)
    # Set up title
    flower_num = image_path.split('/')[1]
    title_ = label_map[flower_num]
    # Plot flower
    img = process_image(image_path)
    imshow(img, ax, title = title_);
    # Make prediction
    probs, labs, flowers = predict(image_path, model) 
    # Plot bar chart
    plt.subplot(2,1,2)
    sns.barplot(x=probs, y=flowers, color=sns.color_palette()[0]);
    plt.show()
```


```python
image_path = 'test/28/image_05230.jpg'
plot_solution(image_path, model)
```


```python
image_path = 'test/1/image_06743.jpg'
plot_solution(image_path, model)
```

## Reminder for Workspace users
If your network becomes very large when saved as a checkpoint, there might be issues with saving backups in your workspace. You should reduce the size of your hidden layers and train again. 
    
We strongly encourage you to delete these large interim files and directories before navigating to another page or closing the browser tab.


```python
# TODO remove .pth files or move it to a temporary `~/opt` directory in this Workspace

import os
checkpoint_dir = '.'  # Replace with the actual directory where your checkpoint files are located
checkpoint_files = [file for file in os.listdir(checkpoint_dir) if file.endswith('.pth')]
print(checkpoint_files)
```


```python

```
