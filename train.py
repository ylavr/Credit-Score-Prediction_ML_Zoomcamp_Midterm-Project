import xgboost as xgb
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import label_binarize
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction import DictVectorizer
import pandas as pd
import logging
import pickle  # For saving the model


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Example: Log the start of the script
logging.info("Starting the training script...")

def load_data():
    # Load cleaned data
    df = pd.read_csv('cleaned_df.csv')

    # Split data into train, validation, and test sets
    df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['credit_score'])
    df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=42, stratify=df_full_train['credit_score'])

    # Save the test dataset for later use
    df_test.to_csv("df_test.csv", index=False)

    # Reset indices for all datasets
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    # Encode target variable
    le = LabelEncoder()
    y_train = le.fit_transform(df_train['credit_score'])
    y_val = le.transform(df_val['credit_score'])
    y_test = le.transform(df_test['credit_score'])

    # Drop target variable from features
    df_train = df_train.drop(columns=['credit_score'])
    df_val = df_val.drop(columns=['credit_score'])
    df_test = df_test.drop(columns=['credit_score'])

    # Convert features to dictionaries for DictVectorizer
    train_dicts = df_train.fillna(0).to_dict(orient='records')
    val_dicts = df_val.fillna(0).to_dict(orient='records')
    test_dicts = df_test.fillna(0).to_dict(orient='records')

    # Initialize and fit DictVectorizer
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(train_dicts)
    X_val = dv.transform(val_dicts)
    X_test = dv.transform(test_dicts)

    # Return processed data
    return X_train, X_val, X_test, y_train, y_val, y_test, le, dv

# Inside `load_data()`
logging.info("Loading and preprocessing data...")
# Load data
X_train, X_val, X_test, y_train, y_val, y_test, label_encoder, dict_vectorizer = load_data()
logging.info(f"Data loaded: X_train shape: {X_train.shape}, y_train length: {len(y_train)}")


# Binarize y_val for multiclass ROC AUC
y_val_binarized = label_binarize(y_val, classes=np.unique(y_train))

# Final model with best parameters
final_model = XGBClassifier(
    learning_rate=0.3,  
    n_estimators=100,    
    max_depth=12, 
    min_child_weight=20,
    objective='multi:softprob',
    num_class= 3,
    random_state=42
)


# Log model training
logging.info("Initializing and training the model...")
# Train the model
final_model.fit(
    X_train, 
    y_train, 
    eval_set=[(X_val, y_val)], 
    verbose=False
)
logging.info("Model training completed.")


# Log evaluation
logging.info("Evaluating the model on validation data...")
# Evaluate the model on the validation set
y_pred_val = final_model.predict(X_val)
y_pred_proba = final_model.predict_proba(X_val) 
logging.info("Model evaluation completed.")


# Calculate AUC
roc_auc = roc_auc_score(y_val_binarized, y_pred_proba, multi_class='ovr')

print("Validation Classification Report:")
print(classification_report(y_val, y_pred_val, target_names=label_encoder.classes_))
print(f"Validation ROC AUC: {roc_auc:.4f}")


# Save the model
logging.info("Saving the model and label encoder to files...")
# Save the trained model and label encoder to files
with open("final_model.pkl", "wb") as f:
    pickle.dump(final_model, f)
    
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

with open("dict_vectorizer.pkl", "wb") as f:
    pickle.dump(dict_vectorizer, f)
logging.info("Model and label encoder,dict vectorizer saved successfully.")


print("Model and label encoder,dict vectorizer saved successfully.")