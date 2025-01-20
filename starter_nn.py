import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


class ThermalDataset(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)
        self.scaler = MinMaxScaler()

        # Fill missing data
        df[['Start_Time', 'Start_Temp', 'End_Time', 'End_Temp']] = df[
            ['Start_Time', 'Start_Temp', 'End_Time', 'End_Temp']].ffill()

        # Fit and transform temperature columns, but exclude time
        self.scaler.fit(df[['T_min (C)', 'T_max (C)', 'T_ave (C)']])
        df[['T_min (C)', 'T_max (C)', 'T_ave (C)']] = self.scaler.transform(df[['T_min (C)', 'T_max (C)', 'T_ave (C)']])

        grouped = df.groupby(['Start_Time', 'Start_Temp', 'End_Time', 'End_Temp'])

        self.X, self.Y, self.time_values = [], [], []

        for _, group in grouped:
            if len(group) == 100:
                # Flatten the input sequence into a single vector
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


# Define Basic Neural Network Model
class BasicNNModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=3):
        super(BasicNNModel, self).__init__()
        self.hidden_layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            *[nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU()) for _ in range(num_layers - 1)],
        )
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Process input through hidden layers
        out = self.hidden_layers(x)
        # Output layer
        out = self.output_layer(out)
        return out


# Training the Model
def train_model():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Specify training and testing datasets
    train_files = [f"ThermalAITest{i}.csv" for i in range(1, 10)]
    test_files = [f"ThermalAITest{i}.csv" for i in range(10, 15)]

    train_paths = [os.path.join(script_dir, "clean_data", file) for file in train_files]
    test_paths = [os.path.join(script_dir, "clean_data", file) for file in test_files]

    # Load training datasets
    train_datasets = [ThermalDataset(path) for path in train_paths]
    combined_train_dataset = ConcatDataset(train_datasets)

    # Load testing datasets
    test_datasets = [ThermalDataset(path) for path in test_paths]

    # Prepare DataLoader for training
    train_dataloader = DataLoader(combined_train_dataset, batch_size=4, shuffle=True)

    # Determine input and output sizes dynamically
    for dataset in train_dataloader.dataset.datasets:
        if len(dataset) > 0:
            input_size = dataset[0][0].shape[0]  # Input size from the first sample
            output_size = dataset[0][1].shape[0]  # Output size from the first sample
            break
    else:
        raise ValueError("All datasets are empty. Check the dataset loading process.")

    # Device and Model Initialization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BasicNNModel(input_size=input_size, hidden_size=64, num_layers=2, output_size=output_size).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training Parameters
    num_epochs = 8000
    patience = 50
    best_loss = float("inf")
    counter = 0

    # Training Loop
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        for inputs, targets, _ in train_dataloader:  # Ignore time values
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= len(train_dataloader)

        # Early Stopping Logic
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}. No improvement in {patience} epochs.")
                break

        # Log every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    return model, test_datasets



# Run the Training
trained_model, test_datasets = train_model()

# Testing Function Placeholder
import matplotlib.pyplot as plt
import numpy as np

def test_model(model, test_datasets):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()  # Set model to evaluation mode

    for i, test_dataset in enumerate(test_datasets):
        dataset_name = f"ThermalAITest{i + 10}.csv"
        print(f"Testing on dataset: {dataset_name}")

        try:
            # Iterate over all examples in the test dataset
            for idx in range(len(test_dataset)):
                test_input, test_actual, time_values = test_dataset[idx]
                test_input = test_input.unsqueeze(0).to(device)  # Add batch dimension

                # Generate predictions
                with torch.no_grad():
                    predicted_output = model(test_input).cpu().numpy()

                # Reshape predictions and actual values
                num_time_steps = time_values.shape[0]
                predicted_output = predicted_output.reshape(num_time_steps, -1)  # Reshape to (99, 3)
                actual_values = test_actual.numpy().reshape(num_time_steps, -1)  # Reshape to (99, 3)

                # Reverse scaling
                scaler = test_dataset.scaler
                predicted_original = scaler.inverse_transform(predicted_output)
                actual_original = scaler.inverse_transform(actual_values)

                # Plot each feature (T_min, T_max, T_ave)
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


# Run the Testing
test_model(trained_model, test_datasets)
