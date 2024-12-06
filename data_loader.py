import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def preprocess_data(df):
    X = df[['TV', 'Radio', 'Newspaper']]
    y = df['Sales']
    return train_test_split(X, y, test_size=0.2, random_state=42)
