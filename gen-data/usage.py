from DataBuilder import DataBuilder

# How To Use:
# This script demonstrates how to use the DataBuilder 
# class to generate random datasets based on predefined 
# categories or custom configurations.


# Example usage:
builder = DataBuilder(category='Nature', rows=5)
df = builder.generate_data()
print("Generated DataFrame:\n", df)

# Custom configuration
custom_config = {
    'columns': {
        'temperature': {'type': 'normal', 'loc': 20, 'scale': 5},
        'humidity': {'type': 'uniform', 'low': 0, 'high': 100},
        'city': {'type': 'choice', 'values': ['NY', 'LA', 'SF']}
    }
}
builder = DataBuilder(category='Custom', rows=5, custom_config=custom_config)
df = builder.generate_data()
print("Generated DataFrame with custom config:\n", df)

# Example of generating correlated features and exporting data
builder = DataBuilder(category='Statistics', rows=5)
df = builder.generate_data()
print("Generated DataFrame with Statistics category:\n", df)
df = builder.add_correlated_feature(df, 'mean', correlation=0.9)
print("DataFrame after adding correlated feature:\n", df)

# Exporting the generated DataFrame to CSV
builder.export_data(df, 'dataset.csv')

# Uniform missing rate
builder = DataBuilder(category='Nature', rows=5, missing_values=True, missing_rate=0.1)
print("Data with uniform missing rate:\n", builder.generate_data())

# Column-specific missing rates
builder = DataBuilder(category='Nature', rows=5, missing_values=True, 
                     missing_rate={'species': 0.1, 'height': 0.3, 'age': 0.05})
print("Data with column-specific missing rates:\n", builder.generate_data())