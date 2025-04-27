# Math547 Final Project

## Human Activity Recognition (HAR) with Smartphones

This project uses the **HAR with Smartphones** dataset from the UCI Repository.  
The dataset includes data from **30 volunteers** who wore a smartphone on their waist to record linear acceleration and angular velocity signals while performing six different physical activities:
- Walking
- Walking Upstairs
- Walking Downstairs
- Sitting
- Standing
- Laying

A total of **561 features** were extracted from the raw signals.

---

### Objective
Predict the physical activity being performed based on the recorded sensor data.

---

### Our Approaches

- **Dimensionality Reduction**
  - **Why**: 561 features is a lot, and many may not be necessary for accurate predictions.
  - **How**: Used scikit-learn libraries to implement PCA, t-SNE, and SelectKBest.

- **Model Comparison**
  - **Why**: Different models, varying in complexity, may yield different results.
  - **How**: 
    - **KNN** and **Multinomial Logistic Regression** using built-in models from scikit-learn.
    - **LSTM** and **Fully Connected Neural Network (FCNN)** built with PyTorch and custom hyperparameters.

  - **Note**: LSTMs are especially effective for this task because they can leverage the time-series nature of the data to capture sequential dependencies in sensor readings.


