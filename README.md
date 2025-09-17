# 📖 About

This project investigates the performance of image classification neural networks leveraging pre-trained models. The main objective is to evaluate the efficiency and accuracy of various architectures in recognizing and categorizing sports images. The analysis compares a neural network developed from scratch to various pre-trained models under diverse training configurations, assessing key metrics such as accuracy and computational complexity.

## Authors

- *Alberto Venturini*
- *Andrea Sciortino*

## Dataset

The [100 Sports Classification dataset](https://www.kaggle.com/datasets/gpiosenka/sports-classification) (available on Kaggle) contains more than 100,000 color images (224×224 pixels) representing 100 different sports categories. For this project, 11 categories were selected, each with 15 to 20 images. 
The dataset is then organized into balanced training, validation, and test sets.

## Project Structure

1. **Data Loading and Augmentation**:
   
   After import raw image data, apply transformations: random cropping, flipping, normalization
   to increase dataset diversity and improve model generalization.

```
# Select a subset of classes
selected_classes = ['baseball', 'boxing', 'basketball', 'bowling',
                    'bull riding' , 'football', 'cheerleading', 'volleyball',
                    'formula 1 racing', 'snow boarding', 'polo']
```
[](images/sport_sample.jpg)

3. **Model Definitions** (**Custom Net**, **DenseNet121**):

   Develop custom convolutional neural network and a pre-trained architecture (with froozen last layers).

6. **Training and Evaluation**:

   Train each model on the processed datasets, monitor performance using validation data, and tune hyperparameters to optimize accuracy and reduce overfitting.

8. **Performance Analysis**. Compare results from different models and configurations by examining metrics like accuracy, precision and training/test loss curves: visualize findings to identify the most effective strategies.

## 💡 Results

DenseNet121, leveraging pre-trained weights and transfer learning, achieved higher accuracy and generalization, particularly with limited training data. The Custom Net demonstrated competitive performance but was more sensitive to the size of the training set and hyperparameter choices. The performance of both the Custom Net and DenseNet121 models was evaluated on the selected sports image dataset using key metrics including accuracy, validation loss, and confusion matrices.







