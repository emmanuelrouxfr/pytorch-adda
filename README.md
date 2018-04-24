# PyTorch-ADDA
 This is a modified PyTorch implementation for [Adversarial Discriminative Domain Adaptation](https://arxiv.org/abs/1702.05464) forked from Corenel's github (https://github.com/corenel/pytorch-adda)

 The aim is to train the source encoder (on the source train set) 50 times and then compute the average and standard deviation of the accuracy when testing it on the target test set.  

## Environment
- Python 3.6
- PyTorch 0.2.0

## Usage

Only training a "src only" model on MNIST and then testing it on USPS (x50)
The following command starts the program:

```shell
python3 main.py
```

## Network

In this experiment, I use three types of network. They are very simple.

- LeNet encoder

  ```
  LeNetEncoder (
    (encoder): Sequential (
      (0): Conv2d(1, 20, kernel_size=(5, 5), stride=(1, 1))
      (1): MaxPool2d (size=(2, 2), stride=(2, 2), dilation=(1, 1))
      (2): ReLU ()
      (3): Conv2d(20, 50, kernel_size=(5, 5), stride=(1, 1))
      (4): Dropout2d (p=0.5)
      (5): MaxPool2d (size=(2, 2), stride=(2, 2), dilation=(1, 1))
      (6): ReLU ()
    )
    (fc1): Linear (800 -> 500)
  )
  ```

- LeNet classifier

  ```
  LeNetClassifier (
    (fc2): Linear (500 -> 10)
  )
  ```

- Discriminator

  ```
  Discriminator (
    (layer): Sequential (
      (0): Linear (500 -> 500)
      (1): ReLU ()
      (2): Linear (500 -> 500)
      (3): ReLU ()
      (4): Linear (500 -> 2)
      (5): LogSoftmax ()
    )
  )
  ```

## I got the following results
|         <td colspan = "2">  </td> | Accuracy when testing on USPS               |
|  setup    | average (over 50 runs) | standard deviation |
| :--------------------------------: | :------------: | :-----------: |
| Source Encoder + Source Classifier |   89.139785%   |  1.501531%   |


## I try to understand why it is different from the results presented in the original branch (https://github.com/corenel/pytorch-adda):

|                                    | USPS (Target) |
| :--------------------------------: | :-----------: |
| Source Encoder + Source Classifier |  83.978495%   |
