import re

def extract_fanout(rtl_code, signal_name):
    pattern = r'\b' + re.escape(signal_name) + r'\b\s*,\s*(\w+)'
    matches = re.findall(pattern, rtl_code)
    return len(matches)

def extract_fanin(rtl_code, signal_name):
    pattern = r'(\w+)\s*,\s*\b' + re.escape(signal_name) + r'\b'
    matches = re.findall(pattern, rtl_code)
    return len(matches)

def extract_signal_type(signal_name):
    if signal_name.startswith('clk') or signal_name.endswith('_clk'):
        return 'clock'
    elif signal_name.startswith('rst') or signal_name.endswith('_rst'):
        return 'reset'
    else:
        return 'data'

def extract_module_depth(rtl_code, signal_name):
    modules = re.findall(r'module\s+(\w+)', rtl_code)
    for i, module in enumerate(modules):
        if re.search(r'\b' + re.escape(signal_name) + r'\b', module):
            return i
    return 0

def extract_features(rtl_file, signal_name):
    with open(rtl_file, 'r') as f:
        rtl_code = f.read()
    
    return {
        'fan_out': extract_fanout(rtl_code, signal_name),
        'fan_in': extract_fanin(rtl_code, signal_name),
        'signal_type': extract_signal_type(signal_name),
        'module_depth': extract_module_depth(rtl_code, signal_name)
    }

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

def prepare_data(csv_file):
    df = pd.read_csv(csv_file)
    X = df[['fan_out', 'fan_in', 'signal_type', 'module_depth']]
    y = df['combinational_depth']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def create_model():
    numeric_features = ['fan_out', 'fan_in', 'module_depth']
    categorical_features = ['signal_type']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first'), categorical_features)
        ])

    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    return model

def train_model(csv_file):
    X_train, X_test, y_train, y_test = prepare_data(csv_file)
    model = create_model()
    model.fit(X_train, y_train)
    
    joblib.dump(model, 'model/rtl_depth_predictor.joblib')
    
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    print(f"Mean Absolute Error: {mae}")
    print(f"Mean Squared Error: {mse}")
    
    return model


from model import train_model

if __name__ == "__main__":
    train_model('data/depth_report.csv')

import joblib
import argparse
from feature_extraction import extract_features
import pandas as pd

def predict_depth(rtl_file, signal_name):
    model = joblib.load('model/rtl_depth_predictor.joblib')
    features = extract_features(rtl_file, signal_name)
    df = pd.DataFrame([features])
    predicted_depth = model.predict(df)
    return predicted_depth[0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict combinational depth.')
    parser.add_argument('--rtl_file', type=str, required=True, help='Path to RTL file')
    parser.add_argument('--signal', type=str, required=True, help='Signal name')
    args = parser.parse_args()
    
    predicted_depth = predict_depth(args.rtl_file, args.signal)
    print(f'Predicted Combinational Depth: {predicted_depth:.2f}')

