import pandas as pandas
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

#  Splitting the features and the target
X = df.drop("Type", axis=1)
y = df["Type"]

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#  Selection of the features
selector = SelectKBest(score_func=f_classif, k=20)
X_selected = selector.fit_transform(X_scaled, y)

# Training and testing data is split in 70:30 ratio
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)