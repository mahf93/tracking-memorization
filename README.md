Code for "Leveraging Unlabeled Data to Track Memorization".

## Requirements
We conducted experiments under:
- python 3.7.10
- torch 1.7.1
- torchvision 0.8.2
- cuda 10.1
- jupyter-notebook 6.1.5
- ipython 7.19.0
- 1 Nvidia Titan X Maxwell GPU

## Description of files
* datasets.py: the code to get data loader for a given batch size, training set size and level of label noise.
* models folder: the folder that contains the neural network configurations that are used.
* experiments.py: the code to train the models and compute susceptibility to noisy labels in each epoch.
* results.ipynb: the code to plot figures after the execution of experiments.py is finished.

## Example to train
To train a resnet on the CIFAR-10 dataset with 50% label noise level, batch size=128, for 200 epochs run the following command:

```
python3 experiments.py --model resnet --filename <filename> --modelfilename <modelfilename>
```
The model is saved in ./checkpoint/ directoty and the results are saved in ./results/ directory.

## A walk-through on a dataset with real-world label noise
To illustrate how to use our method, we provide our results for the Clothing1M dataset [1], which is a real-world dataset with 1M images of clothes. The images have been labeled from the texts that accompany them, hence there are both clean and noisy labels in the set. The ground-truth of the labels is not available (i.e., we do not know which samples have clean labels and which samples have noisy labels). Therefore we cannot explicitly track memorization as measured by the accuracy on a noisy subset of the training set, but we can use susceptibility as a metric, since it does not require access to the ground-truth labels. Although the labels of the training set are not clean, a held-out test set with clean labels is available, in addition to the training set. We do not use this held-out clean set during training, but use it only to evaluate the performance of our approach based on susceptibility.

We train 19 different settings on this dataset with various architectures (ResNet, AlexNet and VGG) and varying learning rates and learning rate schedulers. We compute the training accuracy and susceptibility during the training process for each setting and visualize the results in Figure 1 below.

![My Remote Image](https://i.postimg.cc/dtgthqKt/clothing1m-fig1.png)
*Figure 1*

We divide the models of Figure 1 into 4 regions, where the boundaries are set to the mean value of the training accuracy (horizontal line) and mean value of susceptibility (vertical line): Region 1: Models that are trainable and resistant to memorization, Region 2: Trainable and but not resistant, Region 3: Not trainable but resistant and Region 4: Neither trainable nor resistant. (This is similar to Figure 5 in the paper for CIFAR-10 dataset). This is shown in Figure 2.

![My Remote Image](https://i.postimg.cc/CxjYdkNm/clothing1m-fig2.png)
*Figure 2*


Our approach suggests selecting models in Region 1 (low susceptibility, high training accuracy). 
In order to assess how our approach does in model-selection, we can reveal the test accuracy computed on a held-out clean test set in Figure 3. We observe that the average (± standard deviation) of the test accuracy of models in each region is as follows:

Region 1: **61.799%** &#177; 1.643

Region 2: 57.893% &#177; 3.562

Region 3: 51.250% &#177; 17.209

Region 4: 51.415% &#177; 9.709

![My Remote Image](https://i.postimg.cc/HLYGxKjk/clothing1m-fig3.png)
*Figure 3*


Therefore, our approach selects trainable models with low memorization. Observe that selecting models only on the basis of their training accuracy or only on the basis of their susceptibility fails: both are needed.


[1] Xiao et al., Learning from Massive Noisy Labeled Data for Image Classification, CVPR 2015.


[//]: # "## Access to the paper"

[//]: # "You can find the full version of the paper (including appendices) at ."

[//]: # "## Citation"

[//]: # "To cite our work please use:"

[//]: # "```"

[//]: # "```"
