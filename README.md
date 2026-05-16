# Data Science - Memorial Sloan Kettering Cancer Center Breast Cancer Study 2018 (cBioPortal)
An implementation of Machine Learning and Data Science on a Breast Cancer study from Memorial Sloan Kettering Cancer Center from 2018. This project was part of an intensive 7 week Boston University CS 677 course on Data Science Fundamentals. All data was made available on cBioPortal and I've added it here as well. 

## Instructions
Open the Jupyter Notebook (https://github.com/mentatpsi/BCDataScience/blob/main/Final%20Project%20Continued%20Work.ipynb). 

The notebook above was first implemented a few days after deadline as I couldn't put down the problem.

The whole Notebook should be loadable in browser. This can also be downloaded locally but requires installing Python and Jupyter (can be installed through pip install).

## Limitations
Some of the challenges were overcome through reading more manuals on Machine Learning, and recognizing how to inform Python of data types. While it's not perfect, I was able to resolve limitations of over generalization through changing sample difference in class as well.

## Features
In this study we perform relational data analysis, we perform several machine learning algorithms. We end it with creating a contingency table to find association between gene mutations to possibly help visualize relations across gene networks that were discovered in the decision tree step. The Genes choosen to build the contigency table is non-deterministic, running this locally will produce a different set each time. 

## Purpose
The purpose of this analysis was to help researchers with a collection of tools for data science. It contains query functions, dataframes based on clinical outcome, relational data mapping, and more. I have provided some analysis of the genetics based on small research into each gene mutation. My background is Computer Science, with some minor Computational Biology work. I have highlighted as well some genes that have only been present in each outcome.

I hope this serves a greater purpose to help out researchers.


### Algorithms

#### Random Tree Ensemble
A Random Tree Ensemble is a collection of Decision trees with non-deterministic properties designed to prevent over-fitting through random selection of sample training data and feature sets (columns - in our case gene mutations). They construct a decision tree through usuage of information entropy to enhance the gain of information across a decision node. These trees then form a collection of decision makers that vote on a classification together.

#### Neural Network
The neural network used focuses on a logistic regression activation function, which works through passing a linear regression through a sigmoid function to inspire logical classifcation (binary). In the case of activation, it uses this for propagation throughout the Neural network. We attempt to predict survival based on gene mutations.  

#### Naive Bayes
Naive Bayes employs conditional probabilities and Bayes Theorem to determine class. It is called Naive because it employs the notion of independence to determine class which allows for unique mathematical operations not possible in dependence.


### Continuation on May 2026

I implemented Claude Code to determine statistical significance through Permutation Null Model and Fisher Exact Test for co-mutations. Also added Claude analysis leveraging biology research capabilities.


