import pandas as pd
import seaborn as sns
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


df = pd.read_csv('heart_cleveland_upload.csv')
print(df.shape)
print(df.head())

print(df.info())
print(df.describe())
print(df.isnull().sum())

print(df['condition'].value_counts())

 df['condition'].value_counts().plot(kind='bar', color=['steelblue', 'salmon'])
 plt.title('Class Distribution')
 plt.xlabel('Heart Disease (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()

x = df.drop('condition',axis=1)
y = df['condition']

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

x_train , x_test , y_train , y_test = train_test_split(x,y, test_size=0.2 , random_state=42)

accuracy_list = []

for k in range(1,21):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train,y_train)
    y_pred = knn.predict(x_test)
    accuracy_list.append(accuracy_score(y_test,y_pred))

plt.figure(figsize=(10, 5))
plt.plot(range(1, 21), accuracy_list, marker='o', color='steelblue')
plt.title('KNN Accuracy for Different K Values')
plt.xlabel('K Value')
plt.ylabel('Accuracy')
plt.xticks(range(1, 21))
plt.grid(True)
plt.show()

best_k = accuracy_list.index(max(accuracy_list)) + 1
print(f"Best K = , {best_k}")
print(f"Max Accuracy: {max(accuracy_list)}")

knn_final = KNeighborsClassifier(n_neighbors=best_k)
knn_final.fit(x_train, y_train)
y_pred_knn = knn_final.predict(x_test)

rf = RandomForestClassifier(n_estimators=100 , random_state=42)
rf.fit(x_train,y_train)
y_pred_rf = rf.predict(x_test)

print("=== KNN Model ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred_knn):.2f}")
print(confusion_matrix(y_test, y_pred_knn))
print(classification_report(y_test, y_pred_knn))


print("=== Random Forest Model ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.2f}")
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))


cm = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Disease', 'Disease'],
            yticklabels=['No Disease', 'Disease'])
plt.title('Random Forest Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

feature_importance = pd.Series(rf.feature_importances_, index=x.columns)
 feature_importance.sort_values(ascending=False).plot(kind='bar', color='salmon')
plt.title('Feature Importance - Random Forest')
plt.ylabel('Importance Score')
plt.tight_layout()
plt.show()

joblib.dump(rf, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Saved successfully!")