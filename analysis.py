import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("Breast_cancer_data.csv")

print(df.head())
print(df.columns)

# Split data based on diagnosis
malignant = df[df['diagnosis'] == 0]['mean_radius']
benign = df[df['diagnosis'] == 1]['mean_radius']

# Calculate means
mal_mean = malignant.mean()
ben_mean = benign.mean()
overall_mean = df['mean_radius'].mean()

print("Malignant Mean:", mal_mean)
print("Benign Mean:", ben_mean)
print("Overall Mean:", overall_mean)

# Convert to numpy arrays
mal_array = malignant.to_numpy()
ben_array = benign.to_numpy()

# SSB (Sum of Squares Between Groups)
ssb = (
    len(mal_array) * (mal_mean - overall_mean)**2 +
    len(ben_array) * (ben_mean - overall_mean)**2
)

print("SSB:", ssb)

# SSW (Sum of Squares Within Groups)
ssw = (
    np.sum((mal_array - mal_mean)**2) +
    np.sum((ben_array - ben_mean)**2)
)

print("SSW:", ssw)

# Number of groups
k = 2

# Total observations
N = len(df)

# Degrees of freedom
df_between = k - 1
df_within = N - k

print("DF_between:", df_between)
print("DF_within:", df_within)

# Mean squares
MSB = ssb / df_between
MSW = ssw / df_within

print("MSB:", MSB)
print("MSW:", MSW)

# F-statistic
F_stat = MSB / MSW

print("F-statistic:", F_stat)

# Visualization
sns.boxplot(x="diagnosis", y="mean_radius", data=df)
plt.title("Mean Radius vs Diagnosis")
plt.show()
