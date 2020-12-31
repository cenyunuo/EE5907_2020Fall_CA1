# EE5907_2020Fall
### Background
Here is an [email spam dataset](https://web.stanford.edu/~hastie/ElemStatLearn/datasets/spam.info.txt), consisting of 4601 email messages with 57 features. The data is devided into a 3065 training set and 1536 test set with accompanying labels.

This project is to fit 1) a **Beta-Binomial naive Bayes classifier**, 2) a **Gaussian naive Bayes classifier**, 3) a **logistic regression model** and 4) **a K-Nearest Neighbors classifier** and find the training error and test error.
### Install
This project uses Matlab R2020a. Check if you have any version of [Matlab](https://ww2.mathworks.cn/products/matlab.html).
### Usage
Here is the content of this project,
```
├── Readme.md                   
├── Q1
│   ├── Q1.m
│   └── traning.m
├── Q2
│   ├── Q2.m
│   └── training.m
├── Q3
│   ├── Q3.m
│   ├── newton.m
│   └── training.m
└── Q4
    ├── Q4.m
    ├── euclidean.m
    └── training.m     
```
##### Q1: Beta-Binomial naive Bayes classifier

1. Include `spamData.mat` into the folder `Q1`;
2. Then directly run `Q1.m` in matlab;
3. The code will call `training.m` to training a classifier;
4. The code will plots of training and test error rates versus $\alpha$ and print the training and testing error rates for $\alpha$ = 1, 10 and 100.

##### Q2: Gaussian naive Bayes classifier

1. Include `spamData.mat` into the folder `Q2`;
2. Then directly run `Q2.m` in matlab;
3. The code will automatically call `training.m` to training a classifier;
4. The code will print the training and testing error rates.

##### Q3: Logistic regression model

1. Include `spamData.mat` into the folder `Q3`;
2. Then directly run `Q3.m` in matlab;
3. The code will automatically call `newton.m` and `training.m` to training a classifier;
4. The code will plots of training and test error rates versus $\lambda$ and print the training and testing error rates for $\lambda$ = 1, 10 and 100.

##### Q4: Logistic regression model

1. Include `spamData.mat` into the folder `Q4`;
2. Then directly run `Q4.m` in matlab;
3. The code will automatically call `euclidean.m` and `training.m` to training a classifier;
4. The code will plots of training and test error rates versus K and print the training and testing error rates for K = 1, 10 and 100.
