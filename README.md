# Introduction to Machine Learning and Neural Networks

[![The logo of Electric Square Ltd.](presentation/img/ES-logo-and-wordmark.jpg)](https://www.electricsquare.com/)

Brought to you by [Electric Square](https://www.electricsquare.com/).

Presented by [Tom Read Cutting](https://moosichu.github.io).

## Table of Contents

TODO

## Introduction

- **[Presentation](https://moosichu.github.io/introduction-to-neural-networks/presentation/index.html)**


### Overview

This workshop has 4 goals:

  - Introduce you to the wide and wonderful field of Machine Learning
  - Explain the theory behind simple Neural Networks, one of the most popular machine learning algorithms out there
  - Complete a fun neural network challenge: writing a neural network that can recognise handwritten digits from the [MNIST database](http://yann.lecun.com/exdb/mnist/).
    ![A sample of the MNIST database showing scans of handwritten numbers](presentation/img/MnistExamples.png)
  - Introduce you to a wide array of resources for further learning

### Pre-Requisites

You will need:

  - Python 3
  - A text editor

We will also download these libraries later:

  - [scikit-learn](https://scikit-learn.org/stable/)
  - [NumPy](http://www.numpy.org/)
  - [MNIST-for-Numpy](https://github.com/hsjeong5/MNIST-for-Numpy)


#### Aside: Why Python?

Basically, it is incredibly popular and widely used.

It is an incredibly powerful interpreted language with many useful machine learning and visualisation libraries that can interact well with each other.

Furthermore, Python is incredibly widely used as a scripting language for other applications such as [Blender](https://www.blender.org/) and [Houdini](https://www.sidefx.com/). This allows Python and machine learning libraries to be used to enhance those tools greatly with relatively little effort. (eg. [Houdini](https://www.sidefx.com/) and [scikit-learn](https://scikit-learn.org) were used in conjuction to speed up development of [Spider-Man: Into the Spider-Verse](https://sidefx.com/community/spider-man-into-the-spider-verse/)).

![Example of machine learning being used to assist the 2D inking of 3D Spider-Verse frames](presentation/img/spiderverse-inkline-example.jpg)


## The Background

### What is machine learning?

Machine Learning (ML) is a form of computation where a program is designed to solve specific tasks using models and inference, as opposed to relying on explicit hard-coded instructions.

Whilst fairly easy to define, this doesn't really help understand anything about *how* machine learning works. However, it does help us understand what makes it so *powerful*. Essentially it means that forms of computation exist which allows us to solve problems that *to this day* we wouldn't be able to solve otherwise with explicit instructions. 

No one knows how to write a 'recognize face' algorithm, but we have trained generic machine learning algorithms to recognize faces for us.

### Types of machine learning

To better understand how machine learning works, we can divide it into three broad categories:

 1. Supervised Machine Learning
 2. Unsupervised Machine Learning
 3. Reinforcement Machine Learning

Deciding which one of these can solve a given problem is a good way to narrow down which algorithm you may want to use. However, some systems (such as neural networks) can be adapted for all three!

### Supervised Machine Learning

This is when a machine learning algorithm attempts to infer a function from labeled training data.

TODO: explain example

  - The MNIST Database
  - MariFlow
  - Google Translate
  - TensorFlow Playground

### Unsupervised Machine Learning

This is when a machine learning algorithm attempts to infer some kind underlying structure to unlabelled data. 

TODO: explain example

    Clustering Workout Sessions
    CycleGan
    Super-Resolution
    Edmond de Belamy

### Reinforcement Machine Learning

This is when a machine learning algorithm seeks to take actions in order to maximize some kind of reward.

TODO: explain example


    MarI/O
    Random Network Distillation
    AlphaGo
    Unity ML-Agents Toolkit

## The Theory

Now that we have covered some background and given some basic information about the field of machine learning, lets move onto a concrete example of a machine learning algorithm: Neural Networks.

### An Example of Neural Networks

Although you have probably *heard* of neural networks at some point in your life by now if you are vaguely interested in computers, it's still a good idea to familiarize yourself with what they look like and how they behave before jumping into understanding what is going on under-the-hood.

Go to **[playground.tensorflow.org](https://playground.tensorflow.org)** and have a play around with the Neural Network there to get a feel for how they work. Don't about trying to understand what everything means, just get a feel for what kind of structure that we will be dealing with and then at the end of this workshop you will be able to come back to this page and *really* understand what is happening.


TODO: Continue section

## The Practice

TODO: Adapt from presentation

