# EDA MODULE
# Khám phá dữ liệu và trực quan hóa


import matplotlib.pyplot as plt
import seaborn as sns

def plot_distribution(df, col):
    plt.figure(figsize=(6,4))
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.show()

def correlation_heatmap(df):
    plt.figure(figsize=(10,6))
    sns.heatmap(df.corr(), annot=True, cmap='Blues')
    plt.title('Correlation Heatmap')
    plt.show()