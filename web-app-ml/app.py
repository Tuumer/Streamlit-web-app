# Import necessary libraries
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import learning_curve

# Define the Streamlit app
def main():
    # Title of the app
    st.title('Machine Learning Web Application')

    # Add a sidebar
    st.sidebar.title('User Inputs')

    # Add file uploader for dataset
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

    # Check if a file was uploaded
    if uploaded_file is not None:
        # Read the data
        data = pd.read_csv(uploaded_file)

        # Show the dataset
        st.write('### Dataset')
        st.write(data)

        # Data preprocessing and model training code...
        # Data preprocessing and wrangling techniques:
        # Convert selected columns to numeric
        numeric_columns = ['Трудности со сном (по шкале «1-низкий, 5-высокий»)', 'Дневная усталость ( по шкале «1-низкий, 5-высокий»)', 'Частота потребления кофеина перед сном (1-редко, 5-очень часто)', 'Частота чувства стресса / тревоги перед сном (1-редко, 5-очень часто)']
        data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors='coerce')

        # Handle missing values appropriately for numeric columns
        numeric_df = data.select_dtypes(include=['float64', 'int64'])
        data.fillna(numeric_df.mean(), inplace=True)

        # Fill missing values in non-numeric columns with forward fill
        non_numeric_columns = data.select_dtypes(exclude=['float64', 'int64']).columns
        data[non_numeric_columns] = data[non_numeric_columns].fillna(method='ffill', axis=0)

        # Print shape of DataFrame before removing outliers
        st.write("Shape before removing outliers:", data.shape)

        # Remove outliers using IQR method
        Q1 = data[numeric_columns].quantile(0.25)
        Q3 = data[numeric_columns].quantile(0.75)
        IQR = Q3 - Q1
        data = data[~((data[numeric_columns] < (Q1 - 1.5 * IQR)) | (data[numeric_columns] > (Q3 + 1.5 * IQR))).any(axis=1)]

        # Print shape of DataFrame after removing outliers
        st.write("Shape after removing outliers:", data.shape)

        # One-hot encoding for categorical columns
        data = pd.get_dummies(data)

         # Split the dataset into features and target variable
        X = data.drop(["Трудности со сном (по шкале «1-низкий, 5-высокий»)"], axis=1)  # Features
        y = data["Трудности со сном (по шкале «1-низкий, 5-высокий»)"]  # Target variable

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Define and train the supervised model (RandomForestClassifier)
        st.write('### RandomForestClassifier')
        st.write('RandomForestClassifier is an ensemble learning method used for classification tasks.')
        st.write('It can be used for a wide range of classification problems and works well with complex datasets.')
        rf_params = {
            'n_estimators': [100, 300, 500],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        rf_grid = GridSearchCV(RandomForestClassifier(), rf_params, cv=5, n_jobs=-1)
        rf_grid.fit(X_train, y_train)
        rf_y_pred = rf_grid.predict(X_test)
        # Check for overfitting/underfitting
        st.write('Overfitting/Underfitting Check:')
        st.write(f'Training Accuracy: {rf_grid.best_score_}')
        st.write(f'Testing Accuracy: {accuracy_score(y_test, rf_y_pred)}')
        # Plot learning curve
        plot_learning_curve(rf_grid.best_estimator_, 'RandomForestClassifier Learning Curve', X, y, cv=5)
        # Display the best parameters
        st.write('Best parameters:', rf_grid.best_params_)
        # Display the model evaluation metrics for RandomForestClassifier with hyperparameter tuning
        st.write('Model Evaluation Metrics:')
        st.write('Accuracy:', accuracy_score(y_test, rf_y_pred))
        st.write('Classification Report:')
        st.write(classification_report(y_test, rf_y_pred))
        st.write('Confusion Matrix:')
        st.write(confusion_matrix(y_test, rf_y_pred))

        # Define and train the unsupervised model (KMeans)
        st.write('### KMeans')
        st.write('KMeans is a clustering algorithm used for unsupervised learning tasks.')
        st.write('It is used to partition the dataset into clusters based on similarity.')
        kmeans_params = {
            'n_clusters': [3, 5, 7, 10]
        }
        kmeans_grid = GridSearchCV(KMeans(), kmeans_params, cv=5, n_jobs=-1)
        kmeans_grid.fit(X_train)
        kmeans_y_pred = kmeans_grid.predict(X_test)
        # Plot learning curve
        plot_learning_curve(kmeans_grid.best_estimator_, 'KMeans Learning Curve', X, y, cv=5)
        # Display the best parameters
        st.write('Best parameters:', kmeans_grid.best_params_)

        # Define and train the third model of your choice (e.g., SVC)
        st.write('### SVC')
        st.write('SVC is a classification algorithm that works well with small to medium-sized datasets.')
        st.write('It can handle both linear and non-linear classification tasks.')
        svc_params = {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
        }
        svc_grid = GridSearchCV(SVC(), svc_params, cv=5, n_jobs=-1)
        svc_grid.fit(X_train, y_train)
        svc_y_pred = svc_grid.predict(X_test)
        # Check for overfitting/underfitting
        st.write('Overfitting/Underfitting Check:')
        st.write(f'Training Accuracy: {svc_grid.best_score_}')
        st.write(f'Testing Accuracy: {accuracy_score(y_test, svc_y_pred)}')
        # Plot learning curve
        plot_learning_curve(svc_grid.best_estimator_, 'SVC Learning Curve', X, y, cv=5)
        # Display the best parameters
        st.write('Best parameters:', svc_grid.best_params_)
        # Display the model evaluation metrics for SVC with hyperparameter tuning
        st.write('Model Evaluation Metrics:')
        st.write('Accuracy:', accuracy_score(y_test, svc_y_pred))
        st.write('Classification Report:')
        st.write(classification_report(y_test, svc_y_pred))
        st.write('Confusion Matrix:')
        st.write(confusion_matrix(y_test, svc_y_pred))


def plot_learning_curve(estimator, title, X, y, cv=5):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5))

    # Plot learning curve
    plt.figure()
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.legend(loc="best")

    # Display the plot
    st.pyplot(plt)


if __name__ == '__main__':
    main()

