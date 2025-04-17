from data import get_data
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neighbors import KNeighborsClassifier


# Load activity labels from activity_labels.txt
activity_labels_path = 'data/activity_labels.txt'
activity_labels = pd.read_csv(activity_labels_path, sep=' ', header=None, names=['Label', 'Activity'])

# Create a mapping from activity name to integer label
activity_to_label = dict(zip(activity_labels['Activity'], activity_labels['Label']))

if __name__ == "__main__":
    train, test = get_data()
    # --------------------------
    # 1. Prepare the data
    # --------------------------
    # Assume you have DataFrames: train and test,
    # with 'Activity' as the target label.

    # Features and labels
    X_train = train.drop(columns=['Activity'])
    y_train = train['Activity']
    X_test = test.drop(columns=['Activity'])
    y_test = test['Activity']

    # Map activity names to integer labels
    y_train_mapped = y_train.map(activity_to_label)
    y_test_mapped = y_test.map(activity_to_label)

    # --------------------------
    # 2. Scale the features
    # --------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --------------------------
    # 3. Apply PCA
    # --------------------------
    # Choose the number of components (e.g., 10) or use a variance threshold.
    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    # print("Total explained variance:", sum(pca.explained_variance_ratio_))
    # print("Number of components after PCA:", pca.n_components_)
    # Select the 10 best features using ANOVA F-value

    # selector = SelectKBest(score_func=f_classif, k=300)

    # X_train_best = selector.fit_transform(X_train_scaled, y_train_mapped)
    # X_test_best = selector.transform(X_test_scaled)

    # selected_feature_indices = selector.get_support(indices=True)
    # selected_features = X_train.columns[selected_feature_indices]
    # print("Selected 10 best features:")
    # print(selected_features)
    # X_train_pca = X_train_best
    # X_test_pca = X_test_best

    X_train_pca = X_train_scaled
    X_test_pca = X_test_scaled

    # --------------------------
    # 4. Train a classifier
    # --------------------------
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train_pca, y_train)

    # knn = KNeighborsClassifier(n_neighbors=5)
    # knn.fit(X_train_pca, y_train)
    # --------------------------
    # 5. Evaluate on the test set
    # --------------------------
    y_pred = cfl.predict(X_test_pca)
    # print(y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    # print(classification_report(y_test, y_pred))

    print(f"Test Accuracy with PCA = {accuracy:.2%}")

