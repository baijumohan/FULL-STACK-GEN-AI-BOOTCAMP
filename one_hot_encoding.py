# One-Hot Encoding Example with Dummy Data

from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np

# Create dummy data
data = {
    'Color': ['Red', 'Blue', 'Green', 'Red', 'Blue'],
    'Size': ['Small', 'Medium', 'Large', 'Medium', 'Small'],
    'Shape': ['Circle', 'Square', 'Triangle', 'Circle', 'Square']
}

df = pd.DataFrame(data)
print("=" * 60)
print("ORIGINAL DATA:")
print("=" * 60)
print(df)
print("\n")

# Method 1: Using pandas get_dummies (simpler and recommended for most cases)
print("=" * 60)
print("METHOD 1: Using Pandas get_dummies()")
print("=" * 60)
encoded_df_pandas = pd.get_dummies(df, columns=['Color', 'Size', 'Shape'], dtype=int)
print(encoded_df_pandas)
print("\nShape:", encoded_df_pandas.shape)
print("\n")

# Method 2: Using sklearn OneHotEncoder (more powerful, useful for pipelines)
print("=" * 60)
print("METHOD 2: Using Sklearn OneHotEncoder()")
print("=" * 60)
encoder = OneHotEncoder(sparse_output=False)
encoded_array = encoder.fit_transform(df[['Color', 'Size', 'Shape']])

# Convert to DataFrame for better visualization
encoded_df_sklearn = pd.DataFrame(
    encoded_array,
    columns=encoder.get_feature_names_out(['Color', 'Size', 'Shape'])
)
print(encoded_df_sklearn)
print("\nShape:", encoded_df_sklearn.shape)
