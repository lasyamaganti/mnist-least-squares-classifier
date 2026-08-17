# MNIST Least-Squares Classifier

A MATLAB machine learning project using **least-squares optimization to classify handwritten digits** from the MNIST dataset. The model distinguishes the digit 0 from digits 1–9 and explores how training data and feature representation affect classification performance.

## How It Works

- Constructed feature matrices from MNIST pixel data
- Trained a binary classifier using least-squares optimization
- Evaluated error, false positive, and false negative rates
- Compared performance using 5,000 vs. 100 training images
- Applied random nonlinear feature transformations and tested different feature dimensions
- Visualized learned parameters to identify influential image regions

## Technologies & Concepts

**MATLAB • Machine Learning • Linear Algebra • Least Squares • MNIST • Feature Engineering • Image Classification**

## Key Results

The classifier achieved a **1.94% error rate** when trained on 5,000 images. Reducing the training set to 100 images increased the error rate to **22.72%**, demonstrating the impact of training data on generalization. Random feature transformations further showed how feature dimensionality can affect model performance.

## What I Learned

This project strengthened my understanding of how **linear algebra can be applied to machine learning problems**. I gained experience building and evaluating classifiers, engineering features, analyzing model performance, and exploring the relationship between data quantity, feature representation, and generalization.

## Future Improvements

- Extend the classifier to multi-class digit recognition
- Compare least squares with other classification methods
- Explore additional feature extraction techniques
