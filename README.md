# Fraud-Detection-Tackling-Extreme-Class-Imbalance-Using-Hybrid-Approaches

# Objective
1.	To explore the applicability of VAEs in handling highly imbalanced supervised datasets — investigating how generative modeling can help rebalance data distributions by generating synthetic minority samples.

2.	To compare the performance of VAE-based synthetic data augmentation with traditional imbalanced learning techniques such as weighted logistic regression, Random Forest, Gradient Boosting, XGBoost, and ensemble methods.

3.	To evaluate the effectiveness of VAE-generated data on model performance metrics such as accuracy, precision, recall, F1-score.

# Methodology

Step 1: Identify Minority Class

We first separate the fraudulent transactions from the dataset.
This is because we want the model to learn only from fraud cases in order to generate realistic new fraud samples.

Step 2: Scale Features

All features are standardized (mean = 0, standard deviation = 1).
Scaling helps the VAE train more efficiently and ensures that features with large numeric ranges do not dominate the learning process.

Step 3: Build the Variational Autoencoder (VAE)

The VAE has two parts:

Encoder: Compresses high-dimensional fraud data into a smaller latent vector. This latent vector captures the core characteristics of fraud patterns.

Decoder: Reconstructs the original feature values from the latent vector.

The VAE uses a reparameterization trick to allow random sampling from the latent space while still being trainable.

Step 4: Train the VAE

The VAE is trained only on fraud cases.

The loss function combines two terms:

Reconstruction loss: Measures how accurately the VAE can recreate the original fraud data.

KL divergence: Ensures that the latent vectors follow a standard normal distribution, which allows smooth sampling of new points.

The model gradually learns a compact representation of fraud data and how to generate realistic samples from it.

Step 5: Generate Synthetic Fraud Samples

After training, random points are sampled in the latent space (standard normal distribution).

These points are decoded by the VAE into feature space, producing synthetic fraud transactions.

The generated samples are new, realistic data points, not duplicates of existing fraud cases.

Features are then rescaled back to their original scale to match the original dataset.

Step 6: Combine with Original Non-Fraud Cases

Original non-fraud cases are added to the synthetic fraud samples.

This produces a balanced dataset with roughly equal numbers of fraud and non-fraud transactions.

Step 7: Resulting Dataset

The final dataset contains:

Original non-fraud transactions (~48,974)

Synthetic fraud transactions (~48,000)

The dataset is now balanced, allowing models to learn fraud patterns effectively without being biased toward the majority class.

# Now the data is ready for Model Training

1. Dataset Preparation

After generating synthetic fraud samples using a Variational Autoencoder (VAE), the dataset was balanced:

Non-fraud cases (fraud_flag=0): 48,974

Fraud cases (fraud_flag=1): 48,000

This balanced dataset ensures that the classifier can learn patterns from both classes effectively, avoiding bias toward the majority class.

2. Train-Test Split

The dataset was split into training and testing sets:

Training set: 70% of data

Testing set: 30% of data

This split allows the model to learn patterns on the training set while being evaluated on unseen data.

3. Check for multicollinearity

High multicollinearity among features can cause instability in logistic regression coefficients.
VIF was calculated for all numeric features in the training set.
Features with extremely high VIF values (>10) were removed.

After removing these features, the remaining low-VIF features were used for model training. This step reduces multicollinearity, improves model stability, and ensures that feature coefficients are interpretable.

4. Feature Scaling
Remaining features were standardized using z-score scaling (mean = 0, standard deviation = 1).
Scaling ensures that all features contribute equally to the model and accelerates convergence during training.

5. Logistic Regression Model

Model: Logistic Regression

The model was trained on the scaled low-VIF training data 
and evaluated on the test set.

6. Model Performance

Accuracy:

Training Accuracy: 0.8981

Testing Accuracy: 0.8952

Interpretation of the results :

The model is highly effective at detecting fraud cases (Recall = 0.98 for fraud).
There is a slight reduction in recall for non-fraud cases (Recall = 0.81), but overall balance between precision and recall is excellent.
The high F1-score for both classes indicates that the classifier handles both fraud and non-fraud cases well.

7. Summary

Synthetic fraud samples generated via VAE successfully balanced the dataset.
Feature selection using VIF reduced multicollinearity, improving model stability.
Logistic Regression trained on scaled, low-VIF features achieved ~90% accuracy.
The model demonstrates high recall for fraud detection, which is critical in minimizing financial loss due to undetected fraudulent transactions.

### After this we have compared the results with others models , the results are given in the (comparing other model .ipynb file) [we saw that results are not good in other traditional models]
