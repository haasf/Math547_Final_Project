from data import get_data
from utils import LSTM, eval_model, RNN
import time
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

    

# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = "cpu"
torch.manual_seed(42)  
print(f"Using device: {device}")

# # Load activity labels from activity_labels.txt
activity_labels_path = 'data/activity_labels_LSTM.txt'
activity_labels = pd.read_csv(activity_labels_path, sep=' ', header=None, names=['Label', 'Activity'])

# Create a mapping from activity name to integer label
activity_to_label = dict(zip(activity_labels['Activity'], activity_labels['Label']))

pca_components = [10, 50, 100, 150, 200, 250, 300, 350, 400, 500]  # List of PCA components to test
# pca_components = [1]
results = []

if __name__ == "__main__":
    # Get data 29% test if split is True
    train, test = get_data(split=True)

    train.reset_index(drop=True, inplace=True)
    test.reset_index(drop=True, inplace=True)

    X_train = train.drop(columns=['Activity'])
    y_train = train['Activity']
    X_test = test.drop(columns=['Activity'])
    y_test = test['Activity']


    # Map activity names to their corresponding labels
    y_train_encoded = y_train.map(activity_to_label)
    y_test_encoded = y_test.map(activity_to_label)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)


    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train_encoded, dtype=torch.long).to(device)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test_encoded, dtype=torch.long).to(device)
    
    hidden_size = 64
    output_size = len(activity_to_label)
    num_layers = 1
    num_epochs = 100
    learning_rate = 0.001
    
    for n_components in pca_components:
        pca = PCA(n_components=n_components)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)
        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)

        # Create DataLoader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        # Model parameters
        input_size = X_train_tensor.shape[1]
        model = RNN(input_size, hidden_size, output_size, num_layers).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Early stopping parameters
        patience = 5
        best_test_acc = 0.0
        epochs_without_improvement = 0

        for epoch in range(num_epochs):
            model.train()
            for X_batch, y_batch in train_loader:
                # Add a time dimension (sequence length = 1)
                X_batch = X_batch.unsqueeze(1)

                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            test_acc = eval_model(model, X_test_tensor.unsqueeze(1), y_test_tensor)
            train_acc = eval_model(model, X_train_tensor.unsqueeze(1), y_train_tensor)
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Accuracy: {train_acc:.2f}%, Test Accuracy: {test_acc:.2f}%")

            # Early stopping logic
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                epochs_without_improvement = 0  
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs, Best Test Accuracy: {best_test_acc:.2f}%")
                break
        
        results.append((n_components, best_test_acc))
        print(f"Best Test Accuracy with PCA components {n_components} = {best_test_acc:.2f}%")
        # print(f"Total training time: {time_total:.2f} seconds")
        # print("Training complete.")

    # Print summary of results
    print("\nSummary of Results:")
    for n_components, test_acc in results:
        print(f"PCA Components: {n_components}, Best Test Accuracy: {test_acc:.2f}%")