

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
movies = pd.read_csv(r'C:\Users\nilsm\OneDrive\Skrivbord\BIG data\ml-32m (1)\ml-32m\movies.csv')
reviews = pd.read_csv(r'C:\Users\nilsm\OneDrive\Skrivbord\BIG data\ml-32m (1)\ml-32m\ratings.csv')
tags = pd.read_csv(r'C:\Users\nilsm\OneDrive\Skrivbord\BIG data\ml-32m (1)\ml-32m\tags.csv')
links = pd.read_csv(r'C:\Users\nilsm\OneDrive\Skrivbord\BIG data\ml-32m (1)\ml-32m\links.csv')

print(movies.head())
print(reviews.head())