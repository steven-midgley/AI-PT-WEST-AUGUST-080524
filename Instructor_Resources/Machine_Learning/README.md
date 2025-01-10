# Machine Learning & Deep Learning Resources


### My Machine/Deep Learning Learning Personal Projects
* [Credit Risk Modelling](https://github.com/firobeid/Fintech-machine-learning)
* [Event Driven Trading]())

### Machine Learning for AlgoTrading Book
* [Link to the Github Codes](https://github.com/firobeid/machine-learning-for-trading)


### Essential Reading:
* [A survey of many Clustering Algorithms](https://bobrupakroy.medium.com/a-z-clustering-15ff9684bfe6)
* [K-means Visual Example](https://panel.holoviz.org/)
* [Get Cluster Quality (BESIDES INERTIA)](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html)
* [A Gentle Introduction to Time Series Analysis & Forecasting](https://wandb.ai/iamleonie/A-Gentle-Introduction-to-Time-Series-Analysis-Forecasting/reports/A-Gentle-Introduction-to-Time-Series-Analysis-Forecasting--VmlldzoyNjkxOTMz)
* [Visualize Data in 3D plots!](https://plotly.com/python/3d-scatter-plots/)
* [A Competition Hub for ML Experts for Code Sharing](https://farid.one/kaggle-solutions/)
* [Math behind PCA](https://towardsdatascience.com/pca-clearly-explained-how-when-why-to-use-it-and-feature-importance-a-guide-in-python-7c274582c37e)
* [ML University - Important Concepts](https://mlu-explain.github.io/)
* [ Blend a Porfolio of ML Models for a Meta Model](https://medium.com/geekculture/optimizing-a-portfolio-of-models-f1ed432d728b)
* [Decision Trees](https://towardsdatascience.com/decision-trees-as-you-should-have-learned-them-99862469493e)
* [ORIGIN OF SOME ML ALGORITHMS](https://www.deeplearning.ai/the-batch/issue-146/)
* [Neural Network Playground GUI](https://playground.tensorflow.org/#activation=sigmoid&batchSize=10&dataset=gauss&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=1&seed=0.97170&showTestData=false&discretize=false&percTrainData=40&x=false&y=false&xTimesY=true&xSquared=true&ySquared=true&cosX=false&sinX=true&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false)
* [XGBOOST is all you need](https://xgboosting.com/)
* [ML Algorithms from Scratch](https://github.com/rushter/MLAlgorithms)

### ML System Design
* [ML System Design (Uber)](https://www.uber.com/en-SG/blog/scaling-ai-ml-infrastructure-at-uber/)


### Utility Code / PseudoCode:

   #### Train - Validation - Test Split
   ```python
   from sklearn.model_selection import train_test_split

   # Assuming X is your features and y is your target
   # First split: separate test set (20% of data)
   X_temp, X_test, y_temp, y_test = train_test_split(
      X, y, test_size=0.2, random_state=42
   )

   # Second split: divide remaining data into train and validation (80% train, 20% val of remaining data)
   X_train, X_val, y_train, y_val = train_test_split(
      X_temp, y_temp, test_size=0.25, random_state=42  # 0.25 * 0.8 = 0.2 of original data
   )

   # Final split proportions: 60% train, 20% validation, 20% test
   ```     
   #### Gini Gain

   ```python
   Parent: [60 Yes, 40 No]
   Potential Split:
   Left:  [45 Yes, 15 No]
   Right: [15 Yes, 25 No]

   parentGini = 1 - ((60/100)² + (40/100)²) = 0.48
   leftGini = 1 - ((45/60)² + (15/60)²) = 0.375
   rightGini = 1 - ((15/40)² + (25/40)²) = 0.468
   weightedChildGini = (60/100 × 0.375) + (40/100 × 0.468) = 0.412
   giniGain = 0.48 - 0.412 = 0.068
   ```
   #### R-Squared

   ```python
      ```
      R² = 1 - (SSres / SStot)
      ```
      Where:
      - SSres (Sum of Squares of Residuals) = Σ(yi - ŷi)²
      - SStot (Total Sum of Squares) = Σ(yi - ȳ)²
      - yi = actual values
      - ŷi = predicted values
      - ȳ = mean of actual values
      In expanded form:
      ```
      R² = 1 - [Σ(yi - ŷi)²] / [Σ(yi - ȳ)²]
      ```
      Important notes:
      1. R² ranges from 0 to 1 (can be negative for poor models)
      2. Higher values indicate better fit
      3. 1.0 means perfect prediction
      4. 0.0 means model predicts as well as the mean
   ```

   #### Scaling General Benchmark
   ```python
   StandardScaler (standardization) is best for:

   1.Linear Regression
   2.Logistic Regression
   3.Neural Networks
   4.Gradient Descent-based algorithms
   5.Support Vector Machines (SVM)

   This is because these algorithms assume the input variables are normally distributed and/or use gradient descent optimization. Standardization maintains zero mean and unit variance while preserving the shape of the original distribution.
   MinMaxScaler (normalization) is better for:

   1.Neural Networks (when you need bounded outputs)
   2.k-Nearest Neighbors (KNN)
   3.Neural Networks specifically using sigmoid/tanh activation functions
   4.Algorithms that are sensitive to magnitudes but not distributions

   Some general rules:

   Use StandardScaler when you dont know the distribution of data
   Use MinMaxScaler when you know your data has a bounded range
   Some algorithms like Decision Trees and Random Forests dont require scaling at all
   ```

   #### Remove Redundant Features (Pre-Modelling)

   ```python
   import numpy as np
   from tqdm import tqdm
   #These functions can be added to the pipline
   def calculate_conditions(df_np, missing_threshold):  
      missing = np.mean(np.isnan(df_np), axis=0) > missing_threshold  
      non_variating = np.apply_along_axis(lambda x: len(np.unique(x)) <= 1, axis=0, arr=df_np)  
      return missing, non_variating  
   
   def drop_non_informative_columns(df, missing_threshold=0.99):  
      global missing_non_numeric, non_variating_non_numeric, missing, non_variating
      # Separate numeric and non-numeric columns  
      numeric_cols = df.select_dtypes(include=np.number).columns  
      non_numeric_cols = df.select_dtypes(exclude=np.number).columns  
   
      # Convert DataFrame to NumPy array for numeric columns  
      df_numeric_np = df[numeric_cols].values  
   
      # Compute conditions on NumPy array  
      missing, non_variating = calculate_conditions(df_numeric_np, missing_threshold)  
   
      # Columns to drop for numeric columns  
      drop_columns_numeric = numeric_cols[missing | non_variating]  
   
      # Compute conditions for non-numeric columns  
      missing_non_numeric = df[non_numeric_cols].isna().mean() > missing_threshold  
      non_variating_non_numeric = df[non_numeric_cols].nunique() <= 1  
   
      # Columns to drop for non-numeric columns  
      drop_columns_non_numeric = non_numeric_cols[missing_non_numeric | non_variating_non_numeric]  
   
      # Combine all columns to drop  
      drop_columns = list(drop_columns_numeric) + list(drop_columns_non_numeric)  
   
      # Drop columns from DataFrame  
      df.drop(drop_columns, axis=1, inplace=True)  
   
      return df  

   def get_uninformative_columns(df, threshold=0.95):  
      """  
      Identify columns from a dataframe that have one category which appears more than 'threshold' proportion.  
   
      :param df: Input dataframe  
      :param threshold: Proportion threshold, default 0.95  
      :return: List of uninformative column names  
      """  
      uninformative_columns = []  
      for column in tqdm(df.select_dtypes(include=['object', 'category']).columns):  
         max_proportion = df[column].value_counts(normalize=True).values[0]  
         if max_proportion > threshold:  
               uninformative_columns.append(column)  
      return uninformative_columns 
   ```

   #### Calculate Gini Score
   ```python
   import numpy as np
   from collections import Counter
   import pandas as pd

   def calculate_gini(y):
      '''
      Gini impurity value, which ranges from 0 to (1 - 1/C), where C is the number of classes.
      '''
      if len(y) == 0:
         return 0
      counts = Counter(y)
      total = len(y)
      proportions = [count/total for count in counts.values()]
      return 1 - sum(p*p for p in proportions)

   def calculate_normalized_gini(y):
      '''
      max_gini is calculated as (num_classes - 1) / num_classes, which is the maximum possible Gini impurity for the given number of classes.
      This ensures the Gini index is always between 0 and 1, with 0 indicating perfect purity and 1 indicating maximum impurity.
      '''
      gini = calculate_gini(y)
      unique_values = set(y)
      if len(unique_values) == 1:
         return 0
      max_gini = (len(unique_values) - 1) / len(unique_values)
      return gini / max_gini

   def find_best_split(X, y, feature, verbose=False):
      """
      Find best split point for a feature by checking between all adjacent values.
      """
      # Convert to numpy array and sort
      x_values = X[feature].values
      sort_idx = np.argsort(x_values)
      sorted_x = x_values[sort_idx]
      sorted_y = np.array(y)[sort_idx]
      
      # Get unique values in sorted order
      unique_vals = np.unique(sorted_x)
      
      if len(unique_vals) == 1:
         return None, 0
      
      # Calculate split points between adjacent unique values
      split_points = [(unique_vals[i] + unique_vals[i+1])/2 
                     for i in range(len(unique_vals)-1)]
      
      if verbose:
         print(f"\nFeature '{feature}':")
         print(f"Sorted unique values: {unique_vals}")
         print(f"Possible split points: {split_points}")
      
      best_gain = -float('inf')
      best_split = None
      
      parent_gini = calculate_normalized_gini(sorted_y)
      
      for split in split_points:
         left_mask = sorted_x <= split
         y_left = sorted_y[left_mask]
         y_right = sorted_y[~left_mask]
         
         if len(y_left) == 0 or len(y_right) == 0:
               continue
               
         n_left = len(y_left)
         n_right = len(y_right)
         n_total = n_left + n_right
         
         weighted_child_gini = (
               (n_left/n_total) * calculate_normalized_gini(y_left) +
               (n_right/n_total) * calculate_normalized_gini(y_right)
         )
         
         gain = parent_gini - weighted_child_gini
         
         if verbose:
               print(f"\nTrying split at {split:.2f}:")
               print(f"Left node: {sorted_x[left_mask]} → labels: {y_left}")
               print(f"Right node: {sorted_x[~left_mask]} → labels: {y_right}")
               print(f"Normalized Gini gain: {gain:.4f}")
         
         if gain > best_gain:
               best_gain = gain
               best_split = split
      
      return best_split, best_gain
   # Example usage:
   if __name__ == "__main__":
      # Sample data
      
      # Evaluate all features to find the best split
      feature_splits = {
         feature: find_best_split(X, y, feature, verbose=False)
         for feature in X.columns
      }
      
      best_feature = max(feature_splits.items(), 
                        key=lambda x: x[1][1])  # x[1][1] is the gain
      print(f"Best feature: {best_feature[0]} "
            f"at split {best_feature[1][0]:.2f} "
            f"with gain {best_feature[1][1]:.4f}")
   ```

   #### Custom Metric HyperParam Tuning
   ```
   from sklearn.model_selection import GridSearchCV
   param_grid = {
      'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19],
      'weights': ['uniform', 'distance'],
      'leaf_size': [10, 50, 100, 500]
   }
   grid_clf = GridSearchCV(grid_tuned_model, 
                        param_grid, 
                        verbose=3, 
                        refit='average_precision',
                        scoring={
                              'accuracy': 'accuracy',
                              'precision': 'precision',
                              'recall': 'recall',
                              'f1': 'f1',
                              'average_precision': 'average_precision'
                        }
                        )
   from sklearn.metrics import make_scorer
   def custom_accuracy(y_true, y_pred):
      # Example: weighted accuracy that penalizes false positives more
      fp = sum((y_pred == 1) & (y_true == 0))
      fn = sum((y_pred == 0) & (y_true == 1))
      tp = sum((y_pred == 1) & (y_true == 1))
      tn = sum((y_pred == 0) & (y_true == 0))
      
      return (tp + tn) / (tp + tn + fp * 1.5 + fn)

   # Create custom scorer
   custom_scorer = make_scorer(custom_accuracy)
   grid_clf = GridSearchCV(grid_tuned_model, 
                        param_grid, 
                        verbose=3, 
                        refit=custom_scorer,
                        scoring=custom_scorer
                        )     
   COMMON_METRICS = {
      'Classification': [
         'accuracy',          # Overall accuracy
         'balanced_accuracy', # Balanced accuracy for imbalanced datasets
         'f1',               # F1 score (binary)
         'f1_micro',         # F1 score (multiclass micro-averaged)
         'f1_macro',         # F1 score (multiclass macro-averaged)
         'precision',        # Precision score
         'recall',           # Recall score
         'roc_auc',         # ROC AUC score
         'average_precision' # Average precision score
      ],
      'Regression': [
         'r2',              # R-squared score
         'neg_mean_squared_error',     # Negative mean squared error
         'neg_root_mean_squared_error', # Negative root mean squared error
         'neg_mean_absolute_error',     # Negative mean absolute error
         'neg_median_absolute_error'    # Negative median absolute error
      ]
   }                                    
   ```

   #### ML Pipeline

   ```python
   import pandas as pd
   import numpy as np
   from sklearn.model_selection import train_test_split
   from sklearn.pipeline import Pipeline    
   from typing import Tuple  
   from sklearn.impute import SimpleImputer    
   from sklearn.preprocessing import StandardScaler    
   from sklearn.compose import ColumnTransformer    
   from sklearn.model_selection import train_test_split    
   from xgboost import XGBClassifier    
   import category_encoders as ce   
   from sklearn.base import TransformerMixin  
   from dataclasses import dataclass  
   from sklearn.base import BaseEstimator, ClassifierMixin      
   import gc  
   import xgboost as xgb    
   from sklearn.metrics import roc_curve    
   __author__ = 'Firas Obeid'
   # Setup X and y variables
   # X = df.drop(columns='y')
   # y = df['y'].values.reshape(-1,1)
   # y = np.where(y == 'no', 0, 1)

   # Split the data into training and testing sets
   X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, random_state=42)

   numerical_cols = X.select_dtypes(include=np.number).columns.tolist()  
   categorical_cols = X.select_dtypes(include=['object', 'bool', 'category']).columns.tolist()  

   class ColumnNameKeeper(TransformerMixin):  
      def fit(self, X, y=None):  
         return self  
   
      def transform(self, X):  
         self.column_names = X.columns  
         return X  
      
   class NullImputa(TransformerMixin):      
      def __init__(self, min_count_na = 5):      
         self.min_count_na = min_count_na      
         self.missing_cols = None    
         self.additional_features = []  # Store names of additional features  
         self.column_names = None  # Store column names after transformation  
         
      def fit(self, X, y=None):      
         self.missing_cols = X.columns[X.isnull().any()].tolist()      
         return self      
         
      def transform(self, X, y=None):      
         X = X.copy()  # create a copy of the input DataFrame      
         for col in X.columns.tolist():      
               if col in X.columns[X.dtypes == object].tolist():      
                  X[col] = X[col].fillna(X[col].mode()[0])      
               else:      
                  if col in self.missing_cols:      
                     new_col_name = f'{col}-mi'  
                     X[new_col_name] = X[col].isna().astype(int)     
                     self.additional_features.append(new_col_name)  # Store the new column name  
                  if X[col].isnull().sum() <= self.min_count_na:      
                     X[col] = X[col].fillna(X[col].median())      
                  else:      
                     X[col] = X[col].fillna(-9999)      
         assert X.isna().sum().sum() == 0      
         _ = gc.collect()      
         print("Imputation complete.....") 
         self.column_names = X.columns.tolist()  # Store column names after transformation  
         return X  

      
   def ks_stat(y_pred, dtrain)-> Tuple[str, float]:
      y_true = dtrain.get_label()  
      fpr, tpr, thresholds = roc_curve(y_true, y_pred)  
      ks_stat = max(tpr - fpr)  
      return 'ks_stat', ks_stat 

   class XGBoostClassifierWithEarlyStopping(BaseEstimator, ClassifierMixin):    
      def __init__(self, nfolds=5, **params):    
         self.params = params    
         self.evals_result = {}    
         self.bst = None  
         self.nfolds = nfolds  
         self.cvresult = None  # initialize cvresult attribute 
      
      def fit(self, X, y, **fit_params):    
         dtrain = xgb.DMatrix(X, label=y)  
         self.cvresult = xgb.cv(self.params, dtrain, num_boost_round=10000,  verbose_eval=True, maximize=True,
                           nfold=self.nfolds, metrics=['auc'], custom_metric = ks_stat,
                           early_stopping_rounds=10, stratified=True,  
                           seed=42)  
         self.bst = xgb.train(self.params, dtrain, num_boost_round=self.cvresult.shape[0], feval = ks_stat)  
         return self    
      
      def predict(self, X):    
         dtest = xgb.DMatrix(X)    
         return self.bst.predict(dtest)  

   
   def init_model_params(y_train):
      '''
      “binary:logistic” –logistic regression for binary classification, output probability
      “binary:logitraw” –logistic regression for binary classification, output score before logistic transformation
      '''
      #define class weight dictionary, negative class has 20x weight
      # w = {0:20, 1:1}
      global params
      global plst
      # plst = 
      y_train = pd.DataFrame(y_train)
      params = {"booster" :"gbtree",
               "max_depth" : 5,
               "n_jobs": -1,
               "verbosity" : 3,
               "objective": "binary:logistic",
               "eta": 0.05,
               "colsample_bytree" : 0.3, 
               "tree_method": "exact",
               "scale_pos_weight": int((y_train.value_counts()[0] / y_train.value_counts()[1])),
               "eval_metric": ["auc", "logloss", "error"],
               "subsample" : 0.8, "colsample_bylevel" : 1, "random_state" : 42, "verbosity" : 3}   

   init_model_params(y_train)
   # Create a preprocessor for numerical columns  
   numeric_transformer = Pipeline(steps=[  
      ('custome-imputer', NullImputa(5)),  
      ('scaler', StandardScaler())])  

   # Create a preprocessor for categorical columns  
   categorical_transformer = Pipeline(steps=[  
      ('custome-imputer', NullImputa(5)),  
      ('target_encoder', ce.TargetEncoder())])  
   

   # Combine the preprocessors using a ColumnTransformer  
   preprocessor = ColumnTransformer(  
      transformers=[  
         ('num', numeric_transformer, numerical_cols),  
         ('cat', categorical_transformer, categorical_cols)])  
   
   # Create a pipeline that combines the preprocessor with the estimator  
   pipeline = Pipeline(steps=[('preprocessor', preprocessor),  
                              ('classifier', XGBoostClassifierWithEarlyStopping(**params))])  
   
   # Fit the pipeline to the training data  
   pipeline.fit(X_train, y_train)  
   # classifier__eval_set=[(pipeline.named_steps['preprocessor'].transform(X_test), y_test)])
   # Now you can use pipeline.predict() to make predictions on new data 
   # pipeline.predict(X_holdout) 

   cv_results = pipeline.named_steps['classifier'].cvresult
   cv_results[cv_results.columns[cv_results.columns.str.contains('ks.*mean|mean.*ks', regex=True)]].plot()

   y_train_pred = pipeline.predict(X_train) 
   y_holdout_pred = pipeline.predict(X_holdout) 

   # Fit and transform the data with preprocessor  
   # preprocessed_X_train = pipeline.named_steps['preprocessor'].transform(X_train)  
   # Access the underlying Booster in XGBoost   
   importances = pipeline.named_steps['classifier'].bst.get_score(importance_type='gain')  

   preprocessor = pipeline.named_steps['preprocessor']  

   numeric_column_names_after_preprocessing = preprocessor.transformers_[0][1].named_steps['custome-imputer'].column_names  
   categorical_column_names_after_preprocessing = preprocessor.transformers_[1][1].named_steps['custome-imputer'].column_names  

   features_after_preprocessing = numeric_column_names_after_preprocessing + categorical_column_names_after_preprocessing

   # Map to actual feature names  
   importances_with_feat_names = {features_after_preprocessing[int(feat[1:])]: imp for feat, imp in importances.items()}  
   print("Number of features after preprocessing:", len(features_after_preprocessing))  
   print("Max feature index in importances:", max(int(feat[1:]) for feat in importances.keys()))

   importances = pd.DataFrame(list(importances_with_feat_names.items()), columns=['Feature', 'Importance'])  
   importances['Normalized_Importance'] = importances['Importance'] / importances['Importance'].sum()
   # Sort the DataFrame by importance  
   importances = importances.sort_values(by='Normalized_Importance', ascending=False)  
   ```

### Deep Learning
* [Neural Network Visualization](https://alexlenail.me/NN-SVG/index.html)
   #### Installation Guides for GPU Usage:
      Tensorflow:
      ```markdown
      https://www.tensorflow.org/install/pip#windows-native

      The instructions you shared from the TensorFlow website are for creating a new environment with conda, installing the correct versions of CUDA and cuDNN for TensorFlow, and then installing a version of TensorFlow that is compatible with these. Its important to note that it specifies to install a version of TensorFlow less than 2.11, as versions above 2.10 are not supported on the GPU on Windows Native.

      If you follow these instructions, it should install everything you need to run TensorFlow with GPU support. The final command is a quick check to ensure TensorFlow can detect your GPU.

      Here are the steps: Open your terminal or Anaconda prompt. Create a new conda environment (optional: but recommended to avoid package conflicts). You can do this with conda create -n tf_gpu_env python=3.8, then activate it with conda activate tf_gpu_env. Run the command to install the correct versions of CUDA and cuDNN:

      conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0

      Install a compatible version of TensorFlow:
      python -m pip install "tensorflow<2.11"

      Verify the installation:
      python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

      This should return a list with your GPU if everything is installed correctly.
      ```
      Pytorch:
      ```markdown
      # conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0  
      pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
      https://pytorch.org/get-started/locally/
      ```
   #### Definitions:
      Optimizers:
         * ADAM: Adaptive Moment Estimation: The Adam optimizer, short for “Adaptive Moment Estimation,” is an iterative optimization algorithm used to minimize the loss function during the training of neural networks. Adam can be looked at as a combination of RMSprop and Stochastic Gradient Descent with momentum.
         * SGD: Stochastic Gradient Descent: The SGD optimizer, short for “Stochastic Gradient Descent,” is an iterative optimization algorithm used to minimize the loss function during the training of neural networks. SGD is a simple and effective optimization algorithm that updates the model parameters based on the gradient of the loss function with respect to the parameters.
         * RMSprop: Root Mean Square Propagation: The RMSprop optimizer, short for “Root Mean Square Propagation,” is an iterative optimization algorithm used to minimize the loss function during the training of neural networks. RMSprop is a variant of the AdaGrad optimizer that uses a moving average of the squared gradients to update the model parameters.
         * Adamax: Adaptive Moment Estimation with Infinity Norm: The Adamax optimizer, short for “Adaptive Moment Estimation with Infinity Norm,” is an iterative optimization algorithm used to minimize the loss function during the training of neural networks. Adamax is a variant of the Adam optimizer that uses the infinity norm instead of the L2 norm to update the model parameters.
         AdamW: Adam with Weight Decay Regularization: The AdamW optimizer, short for “Adam with Weight Decay Regularization,” is an iterative optimization algorithm used to minimize the loss function during the training of neural networks. AdamW is a variant of the Adam optimizer that uses weight decay regularization to prevent overfitting.
         * Nadam: Nesterov-accelerated Adaptive Moment Estimation: The Nadam optimizer, short for “Nesterov-accelerated Adaptive Moment Estimation,” is an iterative optimization algorithm used to minimize the loss function during the training of neural networks. Nadam is a variant of the Adam optimizer that uses the Nesterov momentum method to update the model parameters.
         * AdaDelta: Adaptive Delta: The AdaDelta optimizer, short for “Adaptive Delta,” is an iterative optimization algorithm used to minimize the loss function during the training of neural networks. AdaDelta is a variant of the AdaGrad optimizer that uses a moving average of the squared gradients to update the model parameters.

      Loss Functions:
         * Binary Crossentropy: The Binary Crossentropy loss function, short for “Binary Cross-Entropy,” is a loss function used to minimize the loss function during the training of neural networks. Binary Crossentropy is a variant of the Cross-Entropy loss function that is used to minimize the loss function during the training of neural networks.
         * Categorical Crossentropy: The Categorical Crossentropy loss function, short for “Categorical Cross-Entropy,” is a loss function used to minimize the loss function during the training of neural networks. Categorical Crossentropy is a variant of the Cross-Entropy loss function that is used to minimize the loss function during the training of neural networks.
         * sparse_categorical_crossentropy: The sparse_categorical_crossentropy loss function is used when your classes are mutually exclusive, i.e., each sample belongs exactly to one class. In other words, it's used for multi-class classification where each instance can only belong to one class out of many classes, and the classes are represented as integers (0, 1, 2, ..., n).
         In our case today case, we will have two output neurons with a sigmoid activation function, which usually suggests a binary or multi-label problem. However, by using sparse_categorical_crossentropy, we will be treating it as a 2-class classification problem, where each sample can only belong to one class (either class 0 or class 1).
         sparse_categorical_crossentropy is particularly useful when we have a lot of classes, as it saves memory by not having to one-hot encode the class labels.

      Weights/Biases Intializers:
         * glorot_uniform: The Glorot Uniform initializer, short for “Glorot Uniform,” is an initializer used to initialize the weights of a neural network. Glorot Uniform is a variant of the Uniform initializer that is used to initialize the weights of a neural network.
         * he_normal: The He Normal initializer, short for “He Normal,” is an initializer used to initialize the weights of a neural network. He Normal is a variant of the Normal initializer that is used to initialize the weights of a neural network.
         * he_uniform: The He Uniform initializer, short for “He Uniform,” is an initializer used to initialize the weights of a neural network. He Uniform is a variant of the Uniform initializer that is used to initialize the weights of a neural network.
         * lecun_normal: The Lecun Normal initializer, short for “Lecun Normal,” is an initializer used to initialize the weights of a neural network. Lecun Normal is a variant of the Normal initializer that is used to initialize the weights of a neural network.
         * lecun_uniform: The Lecun Uniform initializer, short for “Lecun Uniform,” is an initializer used to initialize the weights of a neural network. Lecun Uniform is a variant of the Uniform initializer that is used to initialize the weights of a neural network.
         * normal: The Normal initializer, short for “Normal,” is an initializer used to initialize the weights of a neural network. Normal is a variant of the Uniform initializer that is used to initialize the weights of a neural network.
         * ones: The Ones initializer, short for “Ones,” is an initializer used to initialize the weights of a neural network. Ones is a variant of the Uniform initializer that is used to initialize the weights of a neural network.
         * random_normal: The Random Normal initializer, short for “Random Normal,” is an initializer used to initialize the weights of a neural network. Random Normal is a variant of the Normal initializer that is used to initialize the weights of a neural network.
         * random_uniform: The Random Uniform initializer, short for “Random Uniform,” is an initializer used to initialize the weights of a neural network. Random Uniform is a variant of the Uniform initializer that is used to initialize the weights of a neural network.
         * uniform: The Uniform initializer, short for “Uniform,” is an initializer used to initialize the weights of a neural network. Uniform is a variant of the Uniform initializer that is used to initialize the weights of a neural network.

         Collaborative filtering (CF):
         * Method to predict a rating for a user-item pair based on the
         history of ratings given by the user and given to the item
         * Most CF algorithms are based on user-item rating matrix
         where each row represents a user, each column an item
         – Entries of this matrix are ratings given by users to items
         * The user-item rating matrix is typically sparse, meaning that most entries are missing (unknown) since users typically only rate a small fraction of available items.

         Restricted Boltzmann Machines (RBMs):
         RBMs are modeled after energy models from the physics domain. The probability that the model assigns to a visible vector (v0) involves a calculation that sums over all the possible hidden vectors(h0s). As you can imagine, when you potentially have millions of features for your visible layer, for computation purposes, this very quickly becomes impractical to carry out.
         Thus, we use an approximation technique tf.random.uniform() computes probability approximations using a method called Gibbs sampling.
         ```python
         def hidden_layer(v0_state, W, hb):
            # probabilities of the hidden units
            h0_prob = tf.nn.sigmoid(tf.matmul([v0_state], W) + hb)
            # sample_h_given_X
            h0_state = tf.nn.relu(tf.sign(h0_prob - tf.random.uniform(tf.shape(h0_prob))))
            return h0_state
         ```
         1. `h0_prob` is a tensor that represents the probabilities of a certain state in the hidden layer of a neural network.
         2. `tf.random.uniform(tf.shape(h0_prob))` generates a tensor with the same shape as h0_prob filled with random values ranging from 0 to 1.
         3. `h0_prob - tf.random.uniform(tf.shape(h0_prob))` subtracts the random tensor from the h0_prob tensor. The result is a tensor of the same shape with values that can be positive (if the corresponding value in h0_prob was larger) or negative (if it was smaller).
         4. `tf.sign(...)` takes the sign of each value in the tensor. This results in a tensor of the same shape filled with -1's (for negative values), 0's (for zeros), and 1's (for positive values).
         5. `tf.nn.relu(...)`  applies the Rectified Linear Unit (ReLU) function to the result tensor. The ReLU function sets all negative values to 0, effectively binarizing the state of each neuron in the hidden layer.
         So, the line of code is generating a binary state of neurons based on their probabilities. The neurons with a probability greater than a random value will be activated (set to 1), while the neurons with a probability less than the random value will be deactivated (set to 0).

         In other words, contrastive divergence method allows us to switch on and off hidden state neurons so that we use there state output to reconstruct the inputs.

         * RBM is an unsupervised learning model that can be used for feature learning and unsupervised pre-training of deep neural networks.
         * It can be used to train deep neural networks, as well as to generate new data samples.

         Image Data Types:
         uint8 has 2^8=256 values 0 to 255, its why the pixel values are 0 to 255thus its a strictly positive number up too 8 bits meaning we have numbers starting at zero ending at 255 (256 numbers)
         
         ```python
         #Converting numeric to image:
         import matplotlib.pyplot as plt
         float_images = [np.array(img).astype(np.uint8) for img in resized_imgs]
         plt.imshow(float_images[0], cmap='gray');
         ```

         Convolutional Neural Networks (CNNs):
         In Convolutional Neural Networks (CNNs), filters are initially often initialized with small random numbers. The specific method of initialization can vary. Some common methods include:
         Glorot/Xavier Initialization: This is a method that initializes the weights with a normal distribution centered on 0 and with variance based on the number of input and output neurons. This method is designed to keep the scale of the gradients roughly the same in all layers.
         He Initialization: This is a variant of Glorot/Xavier Initialization, designed specifically for ReLU (Rectified Linear Unit) activation functions. It's similar to Glorot/Xavier Initialization but takes into account that ReLU neurons can be inactive for half of the inputs.
         Random Initialization: This method involves initializing the weights with small random numbers. In TensorFlow, this can be done with tf.random_normal or tf.truncated_normal. The initialization method can have a significant impact on the speed of training and the final performance of the network. Poor initialization can lead to problems such as vanishing/exploding gradients, which can slow down training or cause it to fail entirely.

         Classification:
         Multiclass classification means a classification task with more than two classes; e.g., classify a set of images of fruits which may be oranges, apples, or pears. Multiclass classification makes the assumption that each sample is assigned to one and only one label: a fruit can be either an apple or a pear but not both at the same time.
         Multilabel classification assigns to each sample a set of target labels. This can be thought of as predicting properties of a data-point that are not mutually exclusive, such as topics that are relevant for a document. A text might be about any of religion, politics, finance or education at the same time or none of these.

         * Multi-class vs Binary-class is the question of the number of classes your classifier is modeling. In theory, a binary classifier is much simpler than multi-class problem, so it's useful to make this distinction. For example, Support Vector Machines (SVMs) can trivially learn a hyperplane to separate two classes, but 3 or more classes make the classification problem much more complicated. In the neural networks, we commonly use Sigmoid for binary, but Softmax for multi-class as the last layer of the model.

         * Multi-label vs Single-Label is the question of how many classes any object or example can belong to. In the neural networks, if we need single label, we use a single Softmax layer as the last layer, thus learning a single probability distribution that spans across all classes. If we need multi-label classification, we use multiple Sigmoids on the last layer, thus learning separate distribution for each class.


   #### Miscellaneous Code:
    ```python
   import tensorflow as tf
   tf.keras.backend.clear_session()
   tf.keras.backend.clear_session()
   np.random.seed(42)
   tf.random.set_seed(42)

   gpu_devices = tf.config.list_physical_devices('GPU')

   if gpu_devices:
      print('Using GPU')
      for gpu in gpu_devices[0:2]:
         tf.config.experimental.set_memory_growth(gpu, True)
   else:
      print('Using CPU')
      tf.config.optimizer.set_jit(True)
      print('used: {}% free: {:.2f}GB'.format(psutil.virtual_memory().percent, float(psutil.virtual_memory().free)/1024**3))#@ 

   gpus = tf.config.list_physical_devices('GPU')
   if gpus:
   # Restrict TensorFlow to only use some GPUs
      try:
         tf.config.experimental.set_visible_devices(gpus[0:2], 'GPU')
         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
         print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
      except RuntimeError as e:
      # Visible devices must be set at program startup
         print(e)

   # fine tuning with Grid Search
   def build_classifier(optimizer):
      # first step: create a Sequential object, as a sequence of layers. B/C NN is a sequence of layers.
      nn_model = Sequential()
      # add the first hidden layer
      nn_model.add(Dense(units=5,kernel_initializer='glorot_uniform',
                     activation = 'relu'))
      # add the second hidden layer
      nn_model.add(Dense(units=5,kernel_initializer='glorot_uniform',
                     activation = 'relu'))
      # add the output layer
      nn_model.add(Dense(units=1,kernel_initializer='glorot_uniform',
                     activation = 'sigmoid'))
      # compiling the NN
      nn_model.compile(optimizer=optimizer,loss='binary_crossentropy',metrics=['accuracy'])
      return nn_model

   new_classifier = KerasClassifier(build_fn=build_classifier)

   # create a dictionary of hyper-parameters to optimize
   parameters = {'batch_size':[25,32], 'nb_epoch':[1,2,10],'optimizer':['adam','rmsprop','sgd']}
   grid_search = GridSearchCV(estimator = new_classifier, param_grid = parameters, scoring = 'accuracy', cv=10)
   grid_search = grid_search.fit(scaled_train[int(0.5*len(scaled_train)):],y_train_select.values[int(0.5*len(scaled_train)):])
   # decreased train size for faster results. This is one way to get results faster if runing short on time
   best_parameters = grid_search.best_params_ 
   best_accuracy = grid_search.best_score_

   #Correct way for setting a seed and placing the model on a specific processor
   tf.keras.backend.clear_session()
   np.random.seed(42)
   tf.random.set_seed(42)

   with tf.device('CPU'):
      # Create the Keras Sequential model
      nn_model = tf.keras.models.Sequential()
      
      # Set input nodes to the number of features
      input_nodes = len(X.columns)
      
      # Add our first Dense layer, including the input layer
      nn_model.add(tf.keras.layers.Dense(units=1, activation="relu", input_dim=input_nodes))
      
      # Add the output layer that uses a probability activation function
      nn_model.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))

   #Initialize an Embeddings layer on your own:
   initializer = tf.initializers.RandomUniform(minval=-0.05, maxval=0.05)  
   embedding_layer = tf.keras.layers.Embedding(input_dim=5000, output_dim=64, embeddings_initializer=initializer)  
    ```

### Applications:
* [Text to Image :: Diffusion Models :: Neural Networks](https://lexica.art/?)
* [Another Text to Image :: Diffusion Models :: Neural Networks](https://huggingface.co/spaces/stabilityai/stable-diffusion)
* [Text to Text :: Recurrent Neural Networks :: Neural Networks](https://typeset.io/papers/)
* [Text to Voice + Text to Video :: Reccurent & Convolutional Neural Network :: Neural Networks](https://www.synthesia.io/features/custom-avatar)
* [Text to Actions::Neural Networks](https://openai.com/blog/chatgpt/)
* [Image to Text](https://huggingface.co/blog/idefics2)
* [Data Science Agent](https://labs.google.com/code/dsa?view=readme)
* [Text to SQL](https://ai.google.dev/gemini-api/tutorials/sql-talk)
* [WebSearch](https://www.blackbox.ai/docs?share-code=GA719xy)



### NLP
   ```python
   1. Tokenize
   2. Lower case all tokens
   3. Remove stop words
   4. Stem the tokens
   5. Lemmatize the tokens
   ```
   #### Tokenization:
      * Sentence Tokenization is best for:

      Document summarization - when you need to identify and extract key sentences
      Question-answering systems - to break down long documents into manageable units for finding relevant answers
      Machine translation - many translation systems work better when processing one sentence at a time
      Readability analysis - calculating metrics like average sentence length
      Topic segmentation - identifying where topics change across sentence boundaries

      * Word Tokenization is better for:

      Text classification - when you need word frequencies or bag-of-words features
      Sentiment analysis - analyzing individual words and their emotional content
      Named Entity Recognition - identifying person names, locations, organizations
      Part-of-speech tagging - since this works at the word level
      Building word embeddings - creating vector representations of individual words
      Spelling correction and text normalization

      * Both:
      This maintains both sentence-level context and word-level detail, which is crucial for many advanced NLP tasks like dependency parsing or coreference resolution.

   #### Stemming vs Lemmatization
      * Use Stemming when:
         Speed is important
         You need rough matching (like in search engines)
         Exact dictionary words aren't necessary

      * Use Lemmatization when:
         Accuracy is important
         You need real, meaningful words
         You're doing semantic analysis
         You have the computational resources available

   #### N-grams:
      N-grams are sequences of n consecutive items (usually words or characters) from a text. They're used to capture patterns and context in language. Let me explain their main uses:

      1. Language Modeling and Prediction
      2.Feature Generation for Machine Learning
         Text classification
         Sentiment analysis
         Authorship attribution
      3. Plagiarism Detection
         Compare n-gram overlap between documents
         Longer n-grams (3-5 words) work well for this
      4. Machine Translation
         Help maintain phrase context
         Improve translation quality
      5. Information Retrieval
      Search engines use n-grams for:
         Query suggestion
         Spell checking
         Finding similar documents

      Common n-gram sizes:
      Unigrams (n=1): Individual words
      Bigrams (n=2): Two consecutive words
      Trigrams (n=3): Three consecutive words

      Character n-grams: Used for:
      Handling misspellings
      Working with languages without clear word boundaries
      Dealing with social media text

      The choice of n depends on your task:
      Larger n captures more context but needs more data
      Smaller n is more flexible but might miss important patterns

      #### Word Embeddings:
      Word embeddings are dense vector representations of words that capture semantic relationships between words. They're used to capture the meaning of words and their context in a machine learning model. Here are some common types of word embeddings:

      * GloVe is a word embedding technique that uses a distributed representation of words based on their context. It's based on the co-occurrence of words in a corpus.
      * FastText is a word embedding technique that uses a neural network to learn word representations. It's based on the skip-gram model.
      * BERT is a transformer-based language model that uses a pre-trained word embedding layer. It's
      a transformer-based model that uses a pre-trained word embedding layer. It's a powerful model for text generation and language modeling.
      * Word2Vec is a popular word embedding technique that uses a neural network to learn word representations. It's based on the skip-gram model.

      #### Topic Modelling:
      * LDA is a probabilistic model. It assumes that documents are a mixture of topics and that each word in the document is attributable to one of the document's topics. In this model, the sum of topic probabilities for a given document equals 1 because it represents the total probability distribution of topics for that document. In other words, it shows how likely each topic is to be relevant to the document.
      * NMF, on the other hand, is a linear algebraic model, based on factorizing the (non-negative) document-term matrix into the product of 2 lower-dimensional matrices. It does not have the same probabilistic interpretation as LDA. The components obtained from NMF represent parts of the original features, not probabilities, hence they don't have to sum up to 1.
      Original features are our TF-IDF per token per document.

* [Part of Speech Tagging](https://en.wikipedia.org/wiki/Buffalo_buffalo_Buffalo_buffalo_buffalo_buffalo_Buffalo_buffalo)   
* [Advanced LLM Tokenization(BPE)](https://tiktokenizer.vercel.app/)
* [RNN Explanation for Language Modelling](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
* [Tabulating Text Data Techniques](https://skrub-data.org/stable/auto_examples/02_text_with_string_encoders.html)

### Large Language Models
* [LLMops Database](https://www.zenml.io/llmops-database)
* [Transformers Intro](https://www.youtube.com/watch?v=XfpMkf4rD6E)
* [Learn about GPTs (Low Code)](https://spreadsheets-are-all-you-need.ai/index.html#watch-the-lessons)
* [Gemini](https://github.com/google/generative-ai-docs/blob/main/site/en/gemini-api/docs/get-started/python.ipynb)
* [Goodle Gemini Docs](https://ai.google.dev/gemini-api/docs/get-started/python)
* [RAGs Using Streamlit](https://discuss.streamlit.io/t/april-16-livestream-advanced-rag-techniques/66368?utm_medium=email&_hsenc=p2ANqtz-_KgvGJdvx1Z47zCXIYPSmD4NTwvLdf0mOfkZR9d40BNWX8UI5QBdnRxkEZF7IU89qz6M2xIjCrDDUvqRtnwbuuFDCaxg&_hsmi=304169172&utm_content=304169172&utm_source=hs_email)
* [Gradio Web Apping your LLM](https://huggingface.co/blog/gradio-reload)
* [Comparing GPT-3.5 & GPT-4](https://techcommunity.microsoft.com/t5/ai-azure-ai-services-blog/comparing-gpt-3-5-amp-gpt-4-a-thought-framework-on-when-to-use/ba-p/4088645?lightbox-message-images-4088645=562554iA575A42840947885)
* [Open Vision Language Model](https://huggingface.co/blog/paligemma)
   #### Definitions:
      * Temperature:
      Temperature controls the degree of randomness in token selection. Higher temperatures result in a higher number of candidate tokens from which the next output token is selected, and can produce more diverse results, while lower temperatures have the opposite effect, such that a temperature of 0 results in greedy decoding, selecting the most probable token at each step.
      Temperature doesn't provide any guarantees of randomness, but it can be used to "nudge" the output somewhat.

      * Top-K and top-P: 
      Like temperature, top-K and top-P parameters are also used to control the diversity of the model's output.
      Top-K is a positive integer that defines the number of most probable tokens from which to select the output token. A top-K of 1 selects a single token, performing greedy decoding.
      Top-P defines the probability threshold that, once cumulatively exceeded, tokens stop being selected as candidates. A top-P of 0 is typically equivalent to greedy decoding, and a top-P of 1 typically selects every token in the model's vocabulary.
      When both are supplied, the Gemini API will filter top-K tokens first, then top-P and then finally sample from the candidate tokens using the supplied temperature.

      * Top-K and top-P are mutually exclusive. If both are supplied, the model will filter top-K tokens first, then top-P and then finally sample from the candidate tokens using the supplied temperature.

      * Some commonly used similarity measures are Euclidean distance, cosine similarity, and the Pearson correlation coefficient.
         * Euclidean distance: Measures the distance between two data points in a multidimensional space and determines which objects are similar or dissimilar. The closer and more similar two objects are, the smaller the Euclidean distance between them.
         * Cosine similarity: Measures the cosine of the angle between two vectors in a multidimensional space and takes on a value ranging between -1.0 and 1.0. If the angle between two vectors is small, then they are quite close together spatially, and the value of the cosine similarity will be close to 1.
         * Pearson correlation coefficient: In a previous module, you learned about this coefficient, which measures the linear correlation between two variables. Like cosine similarity, the Pearson correlation coefficient can take on values ranging between -1.0 and 1.0. 
         When calculating the value of the Pearson correlation coefficient for two variables, the closer the number is to 1, the more closely correlated or similar they are considered to be.
      In the context of text analysis (where negative values are usually not applicable because text data is often represented in non-negative vectors), the cosine similarity ranges from 0 to 1.
      ```python
      import math  
      '''
      dot_product function computes the dot product of two vectors.
      magnitude function computes the magnitude (or length) of a vector.
      cosine_similarity function computes the cosine similarity of two vectors by dividing the dot product of the vectors by the product of their magnitudes.
      '''
      def dot_product(v1, v2):  
      return sum(map(lambda x: x[0] * x[1], zip(v1, v2)))  

      def magnitude(vector):  
      return np.sqrt(dot_product(vector, vector))  

      def cosine_similarity(v1, v2):  
      return dot_product(v1, v2) / (magnitude(v1) * magnitude(v2))  
    ```


### LLM Explainability

* [Tokenization Explainer](https://tiktokenizer.vercel.app/)
   - [Full Lectures](https://github.com/karpathy/minbpe/tree/master)
* [Transformers Explainer](https://poloclub.github.io/transformer-explainer/)
   - [Demo](https://www.youtube.com/watch?v=ECR4oAwocjs)
* [Another GPT Explainer](https://bbycroft.net/llm)

   #### BytePerEncoding
```python
def tokenize_word(word, merges, vocabulary, charset, unk_token="<UNK>"):
    word = ' ' + word
    if word in vocabulary:
        return [word]
    tokens = [char if char in charset else unk_token for char in word]

    for left, right in merges:
        i = 0
        while i < len(tokens) - 1:
            if tokens[i:i+2] == [left, right]:
                tokens[i:i+2] = [left + right]
            else:
                i += 1
        return tokens
```

### Some Final Projects:
* [Skin Cancer Detection](https://huggingface.co/spaces/kkruel/skin_cancer_detection_ai)
* [Data Science Tool](https://cgdproject3.streamlit.app/)
* [Video Meeting Summarizer](https://ai-project-3-meeting-summarizer-rttvfhl39dwghbbhc4ywan.streamlit.app/)


### Future Course:
*[ Mining Massive Datasets - Stanford University](https://www.youtube.com/playlist?list=PLLssT5z_DsK9JDLcT8T62VtzwyW9LNepV1)

