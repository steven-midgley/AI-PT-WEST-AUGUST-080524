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


### Deep Learning


### NLP


### Large Language Models
[LLMops Database](https://www.zenml.io/llmops-database)

### LLM Explainability

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

