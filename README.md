# 📖 About

This project investigates the performance of image classification neural networks leveraging pre-trained models. The main objective is to evaluate the efficiency and accuracy of various architectures in recognizing and categorizing sports images. The analysis compares a neural network developed from scratch to various pre-trained models under diverse training configurations, assessing key metrics such as accuracy and computational complexity.

## Authors

- *Alberto Venturini*
- *Andrea Sciortino*

# Dataset

The [100 Sports Classification dataset](https://www.kaggle.com/datasets/gpiosenka/sports-classification) (available on Kaggle) contains more than 100,000 color images (224×224 pixels) representing 100 different sports categories. For this project, 10 categories were selected, each with 15 to 20 images. 
The dataset is then organized into balanced training, validation, and test sets.

## Project Structure

- Data Loading & Augmentation
Import raw image data, apply transformations such as random cropping, flipping, and normalization to increase dataset diversity and improve model generalization.

- Model Definitions (**Custom Net**, **DenseNet121**)
Develop and configure both a custom convolutional neural network and a DenseNet121 architecture (pretrained with last layers froozen) to serve as the primary classification models.

- Training & Evaluation
Train each model on the processed datasets, monitor performance using validation data, and tune hyperparameters to optimize accuracy and reduce overfitting.

- Performance Analysis
Compare results from different models and configurations by examining metrics like accuracy, precision and training/test loss curves: visualize findings to identify the most effective strategies.











