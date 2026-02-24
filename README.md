**# ML-Assignment**
In collaboration with my team members, I completed four tasks that involved implementing machine learning models in real-world applications. These projects provided valuable insights and practical experience applying machine learning techniques to solve complex problems.


**Overview**
This repository contains Python scripts for various machine-learning tasks. Each script demonstrates different aspects of data preprocessing, model training, and evaluation. Below is a brief description of each script and the expected outputs.

**Scripts**

**1.task1:**
**Hypothesis Space:**
       The hypothesis space refers to the set of all possible models or hypotheses that could explain the data. It's the collection of potential functions, models, or rules that a learning algorithm can choose from to make predictions or learn from data.
The hypothesis space is crucial because the goal of learning algorithms is to find the best hypothesis within this space.

**Inductive Bias:**
      It’s the way the algorithm narrows down the vast hypothesis space to make reasonable predictions. Inductive bias refers to the set of assumptions or prior knowledge that a machine learning algorithm uses to make predictions or generalize from the training data to unseen data.

Classification and regression are two main types of supervised learning tasks in machine learning

**Classification:** Classification is a supervised learning task that aims to predict the categorical label or class of an input based on its features.

**Regression:** Regression is a supervised learning task where the goal is to predict a continuous value or numeric quantity based on input data.

**2. task2.py**
    This script performs the following tasks:
    Loads a dataset from a CSV file.
    Preprocesses the data by handling missing values and encoding categorical variables.
    Trains a Random Forest Regressor model to predict vehicle commissions.
    Evaluate the model using Mean Absolute Error (MAE).
    Visualizes the distribution of actual and predicted commissions.
    Generates a scatter plot to compare actual vs. predicted commissions.
    
    Expected Outputs:
    Mean Absolute Error (MAE): A numerical value indicating the model's performance.
    Mean Absolute Error: 0.1708046875000062
    Predicted Commissions: A table showing the predicted commissions for the first few vehicles.
    Predicted CO2 Emissions (Commissions) by Make and Model: A table summarizing the predicted commissions by vehicle make and model.

    Graphs:
    Distribution of Actual and Predicted Commissions.
    Scatter plot of Actual vs. Predicted Commissions.
    Output Graphs:
    Distribution of Actual and Predicted Commissions
    Scatter plot of Actual vs. Predicted Commissions

**3. task3.py**
    This script demonstrates the use of a Naive Bayes classifier for a categorical dataset:
    Encodes categorical data using LabelEncoder.
    Trains a Categorical Naive Bayes model.
    Predicts the class probabilities and the class for a new data point.
   
    Expected Outputs:
    Class Probabilities: Probabilities for each class (e.g., '+' or '-').
    Predicted Class: The predicted class for the new data point.

    Output:
    Class Probabilities:
    Class 0 (-): 0.1627
    Class 1 (+): 0.8373
    Predicted Class: (+)

**3. task4.1.py**
    This script demonstrates polynomial regression:
    Generates synthetic data for tractor age and maintenance cost.
    Fits a polynomial regression model to the data.
    Visualizes the actual data points and the predicted regression curve.

    Expected Outputs:
    Graph: A scatter plot of tractor age vs. maintenance cost with the predicted regression curve.

    Output Graph:
    Tractor Age vs Maintenance Cost

**4. task4.2.py**
    This script extends task4.1.py by evaluating the polynomial regression model:
    Splits the data into training and testing sets.
    Fits a polynomial regression model.
    Evaluate the model using Mean Squared Error (MSE) and R-squared (R²).
    **Output:**
    Mean Squared Error: 8.47223580004552e-24
    R-squared: 1.0
   
    Libraries Required:
**    pandas
    NumPy
    matplotlib
    seaborn
    scikit-learn**
