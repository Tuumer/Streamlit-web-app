# data analysis and wrangling
import pandas as pd
import numpy as np
import keras2onnx

# machine learning
import sklearn as skl
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, learning_curve
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import classification_report,confusion_matrix, precision_score, recall_score, auc,roc_curve,accuracy_score,f1_score,mean_squared_error, r2_score, silhouette_score, calinski_harabasz_score, adjusted_rand_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import learning_curve

# visualization
import matplotlib.pyplot as plt
import seaborn as sns

#exporting model
import joblib

#web app
import streamlit as st

def load_data():
    print("-"*150)
    
    df = pd.read_csv("input/parkinsons.data")

    print("-"*150)

    return df

def model_load(model_name):
   model = joblib.load(f'input/{model_name}.pkl')

   return model

#detecting outlieres
def outliers(df,col):
    Q1=df[col].quantile(0.25)
    Q3=df[col].quantile(0.75)
    IQR=Q3-Q1
    
    lower_bound=Q1 - 1.5 * IQR
    upper_bound=Q3 + 1.5 * IQR
    
    ls=df.index[(df[col]<lower_bound) | (df[col]>upper_bound) ]
    
    return ls

#function for removing outliers
def remove(df, ls):
    ls=sorted(set(ls))
    df=df.drop(ls)

    return df

def make_prediction(model_name, params):
    # Load the trained model
    model = model_load(model_name)

    # Apply label encoding to the parameters

    # Convert the dictionary of parameters to a DataFrame with a single row
    params_df = pd.DataFrame(params, index=[0])

    # Make predictions with the model
    prediction = model.predict(params_df)

    return prediction

def model_buildilng_lr(df):
   
   X=df.drop(['status'],axis=1)
   y=df["status"]

   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

   minmax = MinMaxScaler()
   X_train_scaled = minmax.fit_transform(X_train)
   X_test_scaled = minmax.transform(X_test)

   LR = LogisticRegression()
   LR.fit(X_train_scaled,y_train)
   y_pred_LR = LR.predict(X_test_scaled)
   print('-'*80)
   print("Logistic Regression :")
   print("-"*16)
   
   return LR, X_train_scaled, X_test_scaled, y_train, y_test

def model_buildilng_svm():
   df_cat = labels_encode()
   X=df_cat.drop(['Испытывали ли вы проблемы'],axis=1)
   y=df_cat["Испытывали ли вы проблемы"]

   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

   minmax = MinMaxScaler()
   X_train_scaled = minmax.fit_transform(X_train)
   X_test_scaled = minmax.transform(X_test)

   SVM = SVC(probability=True, kernel = 'linear')
   SVM.fit(X_train_scaled,y_train)
   y_pred_SVM = SVM.predict(X_test_scaled)
   print('-'*80)
   print("Support Vector Machine:")
   print("-"*16)
   
   return SVM, X_train_scaled, X_test_scaled, y_train, y_test

def model_buildilng_kmeans(X):
    
    # Create KMeans instance
    kmeans = KMeans(n_clusters=3, random_state=42)

    # Fit the model to the data (using only features, without the target variable)
    kmeans.fit(X)

    # Predict the cluster labels
    labels = kmeans.labels_

    # Get cluster centers
    centers = kmeans.cluster_centers_



    silhouette = silhouette_score(X, labels)
    print("Silhouette Score:", silhouette)

    calinski_harabasz = calinski_harabasz_score(X, labels)
    print("Calinski-Harabasz Score:", calinski_harabasz)

    evaluation_results = {
        "Metric": ["Silhouette Score", "Calinski-Harabasz Score"],
        "Score": [silhouette, calinski_harabasz]
    }

    evaluation_df = pd.DataFrame(evaluation_results)
    st.write("Evaluation Metrics:")
    st.table(evaluation_df)

    # Visualize the clusters
    plt.figure(figsize=(7, 5))
    plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels, cmap='viridis', alpha=0.5, edgecolors='k')
    plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200, label='Centroids')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('K-means Clustering')
    plt.legend()
    plt.grid(True)
    plt.show()
    st.pyplot(plt,use_container_width=True )

    return kmeans


def Evaluate_Performance(Model, Xtrain, Xtest, Ytrain, Ytest):
    # Fit the model
    Model.fit(Xtrain, Ytrain)
    
    # Calculate training accuracy
    train_accuracy = Model.score(Xtrain, Ytrain)
    
    # Calculate cross-validation score
    overall_score = cross_val_score(Model, Xtrain, Ytrain, cv=10)
    model_score = np.average(overall_score)
    
    # Predict on test set
    Ypredicted = Model.predict(Xtest)
    
    # Calculate evaluation metrics
    accuracy = accuracy_score(Ytest, Ypredicted)
    precision = precision_score(Ytest, Ypredicted, average='weighted')
    recall = recall_score(Ytest, Ypredicted, average='weighted')
    f1 = f1_score(Ytest, Ypredicted, average='weighted')
    mse = mean_squared_error(Ytest, Ypredicted)
    r2 = r2_score(Ytest, Ypredicted)

    evaluation_results = {
        "Metric": ["Training Accuracy", "Cross Validation Score", "Testing Accuracy", "Precision", "Recall", "F1-Score", "Mean Squared Error (MSE)", "R-squared (R2)"],
        "Score": [round(train_accuracy * 100, 2), round(model_score * 100, 2), round(accuracy * 100, 2), round(precision * 100, 2), round(recall * 100, 2), round(f1 * 100, 2), mse, r2]
    }

    evaluation_df = pd.DataFrame(evaluation_results)
    st.write("Evaluation Metrics:")
    st.table(evaluation_df)
    
    # Print evaluation metrics
    print("\n • Training Accuracy Score : ", round(train_accuracy * 100, 2))
    print(f" • Cross Validation Score : {round(model_score * 100, 2)}")
    print(" ❖ Testing Accuracy Score : ", round(accuracy * 100, 2))
    print(' • Precision Score is :', round(precision * 100, 2))
    print(' • Recall Score is :', round(recall * 100, 2))
    print(' • F1-Score Score is :', round(f1 * 100, 2))
    print(" • Mean Squared Error (MSE):", mse)
    print(" • R-squared (R2) Score:", r2)
    print('-' * 80)
    
    # Detect overfitting/underfitting
    if train_accuracy > accuracy:
        print("Model is potentially overfitting.")
        st.write("Model is potentially overfitting.")
    elif train_accuracy < accuracy:
        print("Model is potentially underfitting.")
        st.write("Model is potentially underfitting.")
    else:
        print("Model is performing consistently on training and testing data.")
        st.write("Model is performing consistently on training and testing data.")
    
    # Plot confusion matrix
    conf_matrix = confusion_matrix(Ytest, Ypredicted)
    plt.figure(figsize=(2, 1))  # Set figure size
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap=plt.cm.Blues, annot_kws={"size": 8})
    plt.title('Predicted Labels', y=1.05, fontsize=10, fontfamily='Times New Roman')
    plt.ylabel('True Labels', labelpad=15, fontsize=10, fontfamily='Times New Roman')
    plt.xlabel('Predicted Labels', labelpad=15, fontsize=10, fontfamily='Times New Roman')
    st.pyplot(plt,use_container_width=False ) 

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):

    plt.figure(figsize=[10, 6], dpi=100)
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

def model_tuning():
    
    LR, X_train, X_test, y_train, y_test = model_buildilng()

   # Define the parameter grid to search
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Create a Random Forest classifier
    rf_classifier = RandomForestClassifier(random_state=42)

    # Instantiate GridSearchCV
    grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)

    # Fit the grid search to the data
    grid_search.fit(X_train, y_train)

    # Get the best hyperparameters
    best_params = grid_search.best_params_
    st.write("Best Hyperparameters:", best_params)

    # Get the best model
    best_model = grid_search.best_estimator_

    # Evaluate the best model
    best_model_accuracy = best_model.score(X_test, y_test)
    st.write("Accuracy of Best Model:", best_model_accuracy)
  
def main():
  
  st.set_page_config(page_title="Finals")

  st.markdown("<div style='background-color:#219C90; border-radius:50px;'><h1 style='text-align:center; color:white;'>Final exam</h1></div>",unsafe_allow_html=True)
  df = load_data()

  indexes_list=[]
  for col in ['MDVP:Fo(Hz)','MDVP:Fhi(Hz)']:
    indexes_list.extend(outliers(df, col))

  data_clean=remove(df, indexes_list)
  print(data_clean.shape)
  
  indexes_list=[]
  for col in ['MDVP:Fo(Hz)','MDVP:Fhi(Hz)']:
    indexes_list.extend(outliers(data_clean, col))

  data_clean=remove(data_clean, indexes_list)
  data_clean=data_clean.drop(columns=['name'])
  print(data_clean.shape)

  print(data_clean.info())

  st.write("")
  st.markdown("<h3 style='text-align:center;'>Online survey</h3>",unsafe_allow_html=True)
  
  with st.sidebar:
    st.header("Navigation")
    selected_page = st.selectbox("Choose a page:", ["Home","Logistic Regression", "Support Vector Machine", "Kmeans"])

  with open("style.css") as f:
    st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)

  if selected_page == "Home":
    # Problem Statement 
    st.header("Problem Statement")

    st.write("- To create Machine Learning Web Application by using streamlit and aklearn libraries.")

    data = st.toggle(label="Show dataset")

    if data:
      st.dataframe(df,hide_index=True)

    # Closer look to the data
    st.header("Dataset info")

    params = st.selectbox(label="Select parameter",options=["Shape","Columns","Head","Tail","Description",],index=None)

    if params == None:
      st.write("...")
    elif params == "Shape":
      st.write("Shape of the Dataset :",df.shape)
    elif params == "Columns":
      st.write("Columns in the Dataset :")
      st.dataframe(df.columns,use_container_width=True,hide_index=True)
    elif params == "Description":
      st.write("Description of the Dataset :")
      st.dataframe(df.describe(),use_container_width=True)
    elif params == "Tail":
      st.write("Last 5 rows of the Dataset :")
      st.dataframe(df.tail(),use_container_width=True)
    else:
      st.write("First 5 rows of the Dataset :")
      st.dataframe(df.head(),use_container_width=True)
      

    # Check values of the columns
    st.header("Checking for duplicates/nulls")

    selected_option = st.selectbox(label="Select column",options=["Show missing values","Show duplicates"],index=None)

    if selected_option == None:
      st.write("...")
    elif selected_option == "Show missing values":
      st.write(f"Null values in data:\n")
      st.dataframe(df.isnull().sum(),use_container_width=True)
    else:
      st.write(f"Duplicated rows:\n {df.duplicated().sum()}")
    
    # Visualisations(graphs,charts,plots etc.)
    st.header("Visualization")

    genre = st.radio(
    "Choose what to visualize",
    ["Mean and skewness", "Distibution of feature MDVP:Fo(Hz)", "Distibution of feature MDVP:Fo(Hz) by status","Outliers",
     "After hanlding outliers", "Correlation"])
    
    feature = 'MDVP:Fo(Hz)'

    if genre == "Mean and skewness":
       
       meanData = 'Mean : ' + str(round(df[feature].mean(),4))        # variable to contain mean of the attribute
       skewData = 'Skewness : ' + str(round(df[feature].skew(),4))    # variable to contain skewness of the attribute
       plt.figure(figsize=(10,5))                                         # setting figure size with width = 10 and height = 5
       fig = sns.distplot(df[feature], bins=30, kde=True)             # seaborn distplot to examine distribution of the feature
       plt.title("Distribution of feature : "+feature+" having "+meanData+" and "+skewData)   # setting title of the figure
       plt.show()
       st.pyplot(plt,use_container_width=True)

    elif genre == "Distibution of feature MDVP:Fo(Hz)":
       
       plt.figure(figsize=(10,5))                                         # setting figure size with width = 10 and height = 5
       # seaborn distplot to examine distribution of the feature of healthy patient
       fig = sns.distplot(df[df['status'] == 0][feature], bins=30, kde=True, label='Healthy')
       # seaborn distplot to examine distribution of the feature of Parkinson's patient
       fig = sns.distplot(df[df['status'] == 1][feature], bins=30, kde=True, label='Parkinson\'s')
       plt.legend()
       plt.title("Distribution of feature : "+feature)                    # setting title of the figure
       plt.show()
       st.pyplot(plt,use_container_width=True)

    elif genre == "Distibution of feature MDVP:Fo(Hz) by status":
       
       g = sns.FacetGrid(df, col='status')
       g.map(plt.hist, 'MDVP:Fo(Hz)', bins=20)
       st.pyplot(g,use_container_width=True)

    elif genre == "Outliers":
       
       plt.figure(figsize=(15, 6))
       boxplot = df.boxplot(color='blue')
       plt.setp(boxplot.get_xticklabels(),rotation=-15, fontsize=7)
       
       st.pyplot(plt,use_container_width=True)
    
    elif genre == "After hanlding outliers":
       plt.figure(figsize=(15, 6))
       boxplot = data_clean.boxplot(color='blue')
       plt.setp(boxplot.get_xticklabels(),rotation=-15, fontsize=7)
       
       st.pyplot(plt,use_container_width=True)

    elif genre == "Correlation":
       plt.figure(figsize=[15, 8], dpi=100)
       plt.title("Correlation Graph", fontsize=20)
    
       cmap = sns.color_palette("Blues")
       
       sns.heatmap(data_clean.corr(), annot=True, cmap=cmap)
       
       st.pyplot(plt,use_container_width=True)
       


  elif selected_page == "Logistic Regression":
    st.write("**This is the first supervised model**")
    st.write("Logistic Regression Model Evaluation:")
    LR, X_train_scaled, X_test_scaled, y_train, y_test = model_buildilng_lr(data_clean)

    Evaluate_Performance(LR, X_train_scaled, X_test_scaled, y_train, y_test)
    with st.container():
        st.sidebar.header("Filters")
        input_dict = {}

        for column in data_clean.columns:
          if column != "Index" and column != "status":
              if data_clean[column].dtype == "object":
                  unique_values = data_clean[column].unique()
                  selected_value = st.sidebar.slider(f"{column}", options=unique_values)
                  input_dict[column] = selected_value
              else:
                  min_val = float(data_clean[column].min())
                  max_val = float(data_clean[column].max())
                  default_val = (min_val + max_val) / 2
                  selected_value = st.sidebar.slider(f"{column}", min_value=min_val, max_value=max_val, value=default_val)
                  input_dict[column] = selected_value

    print(input_dict)
   
    # print(data_clean.describe())
    prediction = make_prediction("LR", input_dict)
    print(prediction)
    # st.write("You have", prediction)
    # if prediction == 1:
    #   st.write("You will probably have problems with online platforms")
    # else:
    #   st.write("You probably won't have any problems with the online platform.")\

    if prediction[0] == 0:
      st.write("<span class='predict negative'>You probably dont have parkinson</span>", unsafe_allow_html=True)
    else:
      st.write("<span class='predict positive'>You probably have parkinson</span>", unsafe_allow_html=True)



    if st.button("Show Learning Curve"):
      fig = plot_learning_curve(LR, "Learning Curve (Logistic Regression)", X_train_scaled, y_train, cv=10)
      st.pyplot(fig,use_container_width=True,)
      print("G")

    if st.button("Model tuning with hyperparameters"):
       model_tuning()





    
  
  elif selected_page == "Support Vector Machine":
    st.write("**This is the second supervised model**")
    st.write("Support Vector Machine Model Evaluation:")
    SVM, X_train_scaled, X_test_scaled, y_train, y_test = model_buildilng_svm()

    Evaluate_Performance(SVM, X_train_scaled, X_test_scaled, y_train, y_test)
    with st.container():
        st.sidebar.header("Filters")
        input_dict = {}

        for column in df.columns:
          if column != "Index" and column != "Испытывали ли вы проблемы":
              if df[column].dtype == "object":
                  unique_values = df[column].unique()
                  selected_value = st.sidebar.select_slider(f"{column}", options=unique_values)
                  input_dict[column] = selected_value
              else:
                  min_val = df[column].min()
                  max_val = df[column].max()
                  default_val = (min_val + max_val) / 2
                  selected_value = st.sidebar.slider(f"Select value for {column}", min_value=min_val, max_value=max_val, value=default_val)
                  input_dict[column] = selected_value

    print(apply_label_encoding(input_dict))
    
    prediction = make_prediction("model_svm", input_dict)

    if prediction[0] == 0:
      st.write("<span class='predict negative'>You probably won't have any problems with the online platform</span>", unsafe_allow_html=True)
    else:
      st.write("<span class='predict positive'>You will probably have problems with online platforms</span>", unsafe_allow_html=True)

    model_buildilng()


    if st.button("Show Learning Curve"):
      fig = plot_learning_curve(SVM, "Learning Curve (Support Vector Machine)", X_train_scaled, y_train, cv=10)
      st.pyplot(fig,use_container_width=True,)
      print("G")

    if st.button("Model tuning with hyperparameters"):
       model_tuning()

  elif selected_page == "Kmeans":
    st.write("**This is the first unsupervised model**")
    st.write("Kmeans Model Evaluation:")

    X = labels_encode()
    kmeans_model = model_buildilng_kmeans(X)

    with st.container():
        st.sidebar.header("Filters")
        input_dict = {}

        for column in df.columns:
          if column != "Index" and column != "Испытывали ли вы проблемы":
              if df[column].dtype == "object":
                  unique_values = df[column].unique()
                  selected_value = st.sidebar.select_slider(f"{column}", options=unique_values)
                  input_dict[column] = selected_value
              else:
                  min_val = df[column].min()
                  max_val = df[column].max()
                  default_val = (min_val + max_val) / 2
                  selected_value = st.sidebar.slider(f"Select value for {column}", min_value=min_val, max_value=max_val, value=default_val)
                  input_dict[column] = selected_value

    print(apply_label_encoding(input_dict))
    
    prediction = make_prediction("model_svm", input_dict)

    if prediction[0] == 0:
      st.write("<span class='predict negative'>You probably won't have any problems with the online platform</span>", unsafe_allow_html=True)
    else:
      st.write("<span class='predict positive'>You will probably have problems with online platforms</span>", unsafe_allow_html=True)




  
  
if __name__ == '__main__':
    main()
