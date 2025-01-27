import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Dataset Class
class ThermalDataset(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)
        self.scaler = MinMaxScaler()

        # Fill missing data
        df[['Start_Time', 'Start_Temp', 'End_Time', 'End_Temp']] = df[
            ['Start_Time', 'Start_Temp', 'End_Time', 'End_Temp']
        ].ffill()

        # Fit and transform temperature columns
        self.scaler.fit(df[['T_min (C)', 'T_max (C)', 'T_ave (C)']])
        df[['T_min (C)', 'T_max (C)', 'T_ave (C)']] = self.scaler.transform(
            df[['T_min (C)', 'T_max (C)', 'T_ave (C)']]
        )

        grouped = df.groupby(['Start_Time', 'Start_Temp', 'End_Time', 'End_Temp'])

        self.X, self.Y, self.time_values = [], [], []

        for _, group in grouped:
            if len(group) == 100:
                # Flatten input sequence into a single vector
                input_sequence = group[['Time (s)', 'T_min (C)', 'T_max (C)', 'T_ave (C)']].values[:-1].flatten()
                target_sequence = group[['T_min (C)', 'T_max (C)', 'T_ave (C)']].values[1:].flatten()

                self.X.append(input_sequence)  # Input features
                self.Y.append(target_sequence)  # Target features
                self.time_values.append(group['Time (s)'].values[1:])  # Original time values

        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.Y = torch.tensor(np.array(self.Y), dtype=torch.float32)
        self.time_values = np.array(self.time_values)  # Keep time values as a NumPy array for easier handling

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.time_values[idx]


# Neural Network Model
class BasicNNModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=4, output_size=3):
        super(BasicNNModel, self).__init__()
        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# Training the Model
def train_model():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Specify training and testing datasets
    train_files = [f"ThermalAITest{i}.csv" for i in range(1, 6)]  # Use only 5 training samples
    test_files = [f"ThermalAITest{i}.csv" for i in range(6, 15)]  # Use the rest for testing

    train_paths = [os.path.join(script_dir, "clean_data", file) for file in train_files]
    test_paths = [os.path.join(script_dir, "clean_data", file) for file in test_files]

    # Load datasets
    train_datasets = [ThermalDataset(path) for path in train_paths]
    combined_train_dataset = ConcatDataset(train_datasets)

    test_datasets = [ThermalDataset(path) for path in test_paths]

    # Split training data into training and validation subsets
    train_size = int(0.8 * len(combined_train_dataset))
    val_size = len(combined_train_dataset) - train_size
    train_subset, val_subset = random_split(combined_train_dataset, [train_size, val_size])

    # Prepare DataLoaders
    train_dataloader = DataLoader(train_subset, batch_size=16, shuffle=True)
    val_dataloader = DataLoader(val_subset, batch_size=16, shuffle=False)

    # Determine input and output sizes dynamically
    input_size = train_subset[0][0].shape[0]
    output_size = train_subset[0][1].shape[0]

    # Initialize model, loss, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BasicNNModel(input_size=input_size, hidden_size=128, num_layers=4, output_size=output_size).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    # Training parameters
    num_epochs = 2000
    patience = 100
    best_val_loss = float("inf")
    counter = 0

    # Training Loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0

        for inputs, targets, _ in train_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_dataloader)

        # Validation Loop
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets, _ in val_dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        val_loss /= len(val_dataloader)

        # Early Stopping Logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}. No improvement in {patience} epochs.")
                break

        # Log every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    return model, test_datasets


# Testing the Model
def test_model(model, test_datasets):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    for i, test_dataset in enumerate(test_datasets):
        dataset_name = f"ThermalAITest{i + 6}.csv"
        print(f"Testing on dataset: {dataset_name}")

        try:
            for idx in range(len(test_dataset)):
                test_input, test_actual, time_values = test_dataset[idx]
                test_input = test_input.unsqueeze(0).to(device)

                # Generate predictions
                with torch.no_grad():
                    predicted_output = model(test_input).cpu().numpy()

                # Reshape predictions and actual values
                predicted_output = predicted_output.reshape(-1, 3)
                actual_values = test_actual.numpy().reshape(-1, 3)

                # Reverse scaling
                scaler = test_dataset.scaler
                predicted_original = scaler.inverse_transform(predicted_output)
                actual_original = scaler.inverse_transform(actual_values)

                # Plot results
                plt.figure(figsize=(12, 6))
                for feature_idx, feature_name in enumerate(["T_min", "T_max", "T_ave"]):
                    plt.plot(
                        time_values,
                        actual_original[:, feature_idx],
                        label=f"{feature_name} (Actual)",
                        color=["blue", "red", "green"][feature_idx],
                    )
                    plt.plot(
                        time_values,
                        predicted_original[:, feature_idx],
                        label=f"{feature_name} (Predicted)",
                        linestyle="dashed",
                        color=["blue", "red", "green"][feature_idx],
                    )

                plt.xlabel("Time (s)")
                plt.ylabel("Temperature (°C)")
                plt.legend()
                plt.title(f"Actual vs. Predicted Trends for {dataset_name}")
                plt.show()

        except Exception as e:
            print(f"Error testing on {dataset_name}: {e}")


# Run Training and Testing
trained_model, test_datasets = train_model()
test_model(trained_model, test_datasets)
