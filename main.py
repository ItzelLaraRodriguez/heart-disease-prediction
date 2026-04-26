import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("data/heart.csv")

# =========================
# 📊 VISUALIZATION (ESSENTIAL ONLY)
# =========================

# 1. Target Distribution
plt.figure()
sns.countplot(x="HeartDisease", data=df)
plt.title("Heart Disease Distribution")
plt.show()

# 2. Key Feature vs Target (ONLY strongest)
plt.figure()
sns.countplot(x="ExerciseAngina", hue="HeartDisease", data=df)
plt.title("ExerciseAngina vs Heart Disease")
plt.show()

plt.figure()
sns.countplot(x="ST_Slope", hue="HeartDisease", data=df)
plt.title("ST_Slope vs Heart Disease")
plt.show()

# 3. Data Cleaning Justification 🔥
# RestingBP
plt.figure()
sns.histplot(df["RestingBP"], kde=True)
plt.axvline(0, color='red', linestyle='--')
plt.title("RestingBP Distribution (0 is invalid)")
plt.show()

# Cholesterol
plt.figure()
sns.histplot(df["Cholesterol"], kde=True)
plt.axvline(0, color='red', linestyle='--')
plt.title("Cholesterol Distribution (0 is unrealistic)")
plt.show()

# 4. Heatmap (cleaned + encoded)
df_encoded_temp = pd.get_dummies(df, drop_first=True)

plt.figure(figsize=(10,6))
sns.heatmap(df_encoded_temp.corr(), cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

# 5. Correlation with target (clear insight)
corr_target = df_encoded_temp.corr()["HeartDisease"].sort_values(ascending=False)
print("\nTop Correlations with HeartDisease:")
print(corr_target.head(8))


# =========================
# 🧹 DATA CLEANING
# =========================
df = df[df["RestingBP"] != 0]
df["Cholesterol"] = df["Cholesterol"].replace(0, df["Cholesterol"].median())

# =========================
# ⚙️ PREPROCESSING
# =========================
df = pd.get_dummies(df, drop_first=True)

X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# =========================
# 🤖 MODEL COMPARISON
# =========================
models = {
    "KNN (K=11)": KNeighborsClassifier(n_neighbors=11),
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier()
}

for name, model in models.items():
    print(f"\n{name}")
    print("-" * 30)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print(classification_report(y_test, y_pred))