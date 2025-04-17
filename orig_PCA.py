from data import get_data
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif

# Load activity labels from activity_labels.txt
activity_labels_path = 'data/activity_labels.txt'
activity_labels = pd.read_csv(activity_labels_path, sep=' ', header=None, names=['Label', 'Activity'])

# Create a mapping from activity name to integer label
activity_to_label = dict(zip(activity_labels['Activity'], activity_labels['Label']))

pca_components = [10, 50, 100, 150, 200, 250, 300, 350, 400, 500]  # List of PCA components to test
# pca_components = [1]
results = []

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

    
    for n_components in pca_components:
        # --------------------------
        # 3. Apply PCA
        # --------------------------
        # Choose the number of components (e.g., 10) or use a variance threshold.
        pca = PCA(n_components=n_components)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)


        # X_train_pca = X_train_scaled
        # X_test_pca = X_test_scaled

        # --------------------------
        # 4. Train a classifier
        # --------------------------
        # clf = LogisticRegression(max_iter=1000, random_state=42)
        # clf.fit(X_train_pca, y_train)
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train_pca, y_train)
        

        # --------------------------
        # 5. Evaluate on the test set
        # --------------------------
        y_pred = knn.predict(X_test_pca)
        # print(y_pred)
        accuracy = accuracy_score(y_test, y_pred)*100

        results.append((n_components, accuracy))

        print(f"Best Test Accuracy with PCA components {n_components} = {accuracy:.2f}%")
            # print(f"Total training time: {time_total:.2f} seconds")
            # print("Training complete.")

    # Print summary of results
    print("\nSummary of Results:")
    for n_components, test_acc in results:
        print(f"PCA Components: {n_components}, Best Test Accuracy: {test_acc:.2f}%")
    # print(classification_report(y_test, y_pred))


