"""The code in this document was used to create the plot of longitude and latitude with class labels (Figure 3)."""

import os
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('ForestNetDataset/train.csv', usecols=['merged_label', 'latitude', 'longitude'])
print(df.head())

colors = {
    'Grassland shrubland': 'red',
    'Other': 'blue',
    'Plantation': 'green',
    'Smallholder agriculture': 'orange' 
}

# Create a scatter plot
plt.scatter(df['longitude'], df['latitude'], c=df['merged_label'].map(colors), alpha=0.5)

# Customize the plot
plt.title('Latitude and Longitude w/ Classes')
plt.xlabel('Longitude')
plt.ylabel('Latitude')

# Add a legend based on the defined colors
for label, color in colors.items():
    plt.scatter([], [], c=color, label=f'{label}')

plt.legend()