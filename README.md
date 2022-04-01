# Neural_Network_Charity_Analysis

## Resources
- Anaconda 4.11.0
- Jupyter Notebook 6.0.3
- Python 3.7.6
- Pandas
- Plotly
- scikit-learn
- Data:  charity_data.csv

## Project Overview

With knowledge of machine learning and neural networks, I used the features in the provided dataset to create a binary classifier that is capable of predicting whether applicants will be successful if funded by a charitable organization.

I used a CSV containing more than 34,000 organizations that have received funding over the years. Within this dataset were a number of columns that capture metadata about each organization, such as the following:

- EIN and NAME—Identification columns
- APPLICATION_TYPE—Alphabet Soup application type
- AFFILIATION—Affiliated sector of industry
- CLASSIFICATION—Government organization classification
- USE_CASE—Use case for funding
- ORGANIZATION—Organization type
- STATUS—Active status
- INCOME_AMT—Income classification
- SPECIAL_CONSIDERATIONS—Special consideration for application
- ASK_AMT—Funding amount requested
- IS_SUCCESSFUL—Was the money used effectively


This project consisted of three technical analysis deliverables and a written report as follows:

- Deliverable 1: Preprocessing Data for a Neural Network Model
- Deliverable 2: Compile, Train, and Evaluate the Model
- Deliverable 3: Optimize the Model
- Deliverable 4: A Written Report on the Neural Network Model (README.md)



## Preprocessing Data for a Neural Network Model

Preprocessed the dataset in order to compile, train, and evaluate the neural network model.

The following preprocessing steps have been performed:

- The EIN and NAME columns have been dropped 
- The columns with more than 10 unique values have been grouped together 
- The categorical variables have been encoded using one-hot encoding 
- The preprocessed data is split into features and target arrays 
- The preprocessed data is split into training and testing datasets 
- The numerical values have been standardized using the StandardScaler() module 

![Charity Del One](https://github.com/PatriciaCB1/Neural_Network_Charity_Analysis/blob/main/Images/Charity%20Del%201.png) 

## Compile, Train, and Evaluate the Model

Using knowledge of TensorFlow, I designed a neural network, or deep learning model, to create a binary classification model that can predict if a funded organization will be successful based on the features in the dataset. I had to think about how many inputs there are before determining the number of neurons and layers in your model. Once I completed that step, I compiled, trained, and evaluated the binary classification model to calculate the model’s loss and accuracy.

The neural network model using Tensorflow Keras contains working code that performs the following steps:
- The number of layers, the number of neurons per layer, and activation function are defined 
- An output layer with an activation function is created 
- There is an output for the structure of the model 
- There is an output of the model’s loss and accuracy 
- The model's weights are saved every 5 epochs 
- The results are saved to an HDF5 file 

![Charity Del Two One](https://github.com/PatriciaCB1/Neural_Network_Charity_Analysis/blob/main/Images/Charity%20Del%20Two%20One.png) 

![Charity Del Two Two](https://github.com/PatriciaCB1/Neural_Network_Charity_Analysis/blob/main/Images/Charity%20Del%20Two%20Two.png) 


## Optimize the Model

Using knowledge of TensorFlow, I optimized the model in an effort to achieve a target predictive accuracy higher than 75%. Three attempts were made.

The model is optimized, and the predictive accuracy is increased to over 75%, or there is working code that makes three attempts to increase model performance using the following steps:
    - Additional neurons are added to hidden layers 
    ![More nodes](https://github.com/PatriciaCB1/Neural_Network_Charity_Analysis/blob/main/Images/More%20nodes.png) 

    - Neurons are reduced from hidden layers
    ![Fewer nodes](https://github.com/PatriciaCB1/Neural_Network_Charity_Analysis/blob/main/Images/Fewer%20nodes.png) 

    - Additional hidden layers are added 
    ![Adding another layer](https://github.com/PatriciaCB1/Neural_Network_Charity_Analysis/blob/main/Images/Adding%20Another%20Layer.png)

    - The activation function of hidden layers are changed for optimization 
    ![Changing Activation Function](https://github.com/PatriciaCB1/Neural_Network_Charity_Analysis/blob/main/Images/Changing%20Activation%20Function.png)
   

## Results

Adding more nodes increased the model accuracy from 54% to 69%.  When I decreased the number of nodes the accuracy decreased to 44%.  Adding another layer left accuracy at 54%.  Changing the activation function for the hidden layers from relu to tanh resulted in an accuracy fo 58%, so only marginally higher.  

The model with the highest accuracy at 73% was as follows:


![Final Model with highest accuracy](https://github.com/PatriciaCB1/Neural_Network_Charity_Analysis/blob/main/Images/Final%20model%20with%20highest%20accuracy.png) 

![Final Model with highest accuracy two](https://github.com/PatriciaCB1/Neural_Network_Charity_Analysis/blob/main/Images/Final%20model%20with%20highest%20accuracy%20two.png) 
