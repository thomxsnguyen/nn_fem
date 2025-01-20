import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


# Train: Sample 1, 2, 7

# Load and Prepare Data
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
                self.X.append(group[['Time (s)', 'T_min (C)', 'T_max (C)', 'T_ave (C)']].values[:-1])  # Input sequence
                self.Y.append(group[['T_min (C)', 'T_max (C)', 'T_ave (C)']].values[1:])  # Target sequence
                self.time_values.append(group['Time (s)'].values[1:])  # Original time values

        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.Y = torch.tensor(np.array(self.Y), dtype=torch.float32)
        self.time_values = np.array(self.time_values)  # Keep time values as a NumPy array for easier handling

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.time_values[idx]


# Define LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, num_layers=2, output_size=3):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden and cell states for LSTM
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Pass input through LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Pass the LSTM output through the fully connected layer
        out = self.fc(out)
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

    # Device and Model Initialization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMModel().to(device)
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


def test_model(model, test_datasets):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
                    predicted_sequence = model(test_input).cpu().numpy()

                # Extract predicted values
                t_min_pred = predicted_sequence[0, :, 0]  # Scaled predictions for T_min
                t_max_pred = predicted_sequence[0, :, 1]  # Scaled predictions for T_max
                t_ave_pred = predicted_sequence[0, :, 2]  # Scaled predictions for T_ave

                # Extract actual values
                t_min_actual = test_actual[:, 0].numpy()
                t_max_actual = test_actual[:, 1].numpy()
                t_ave_actual = test_actual[:, 2].numpy()

                # Reverse scaling
                scaler = test_dataset.scaler
                t_min_pred_original = scaler.inverse_transform(
                    np.column_stack((t_min_pred, t_min_pred, t_min_pred))
                )[:, 0]
                t_max_pred_original = scaler.inverse_transform(
                    np.column_stack((t_max_pred, t_max_pred, t_max_pred))
                )[:, 1]
                t_ave_pred_original = scaler.inverse_transform(
                    np.column_stack((t_ave_pred, t_ave_pred, t_ave_pred))
                )[:, 2]

                t_min_actual_original = scaler.inverse_transform(
                    np.column_stack((t_min_actual, t_min_actual, t_min_actual))
                )[:, 0]
                t_max_actual_original = scaler.inverse_transform(
                    np.column_stack((t_max_actual, t_max_actual, t_max_actual))
                )[:, 1]
                t_ave_actual_original = scaler.inverse_transform(
                    np.column_stack((t_ave_actual, t_ave_actual, t_ave_actual))
                )[:, 2]

                # Plot predictions and actual values using the original time values
                plt.figure(figsize=(10, 5))
                plt.plot(time_values, t_min_actual_original, label="T_min (Actual)", color="blue")
                plt.plot(time_values, t_min_pred_original, label="T_min (Predicted)", linestyle="dashed", color="blue")

                plt.plot(time_values, t_max_actual_original, label="T_max (Actual)", color="red")
                plt.plot(time_values, t_max_pred_original, label="T_max (Predicted)", linestyle="dashed", color="red")

                plt.plot(time_values, t_ave_actual_original, label="T_ave (Actual)", color="green")
                plt.plot(time_values, t_ave_pred_original, label="T_ave (Predicted)", linestyle="dashed", color="green")

                plt.xlabel("Time (s)")
                plt.ylabel("Temperature (°C)")
                plt.legend()
                plt.title(f"Actual vs. Predicted Trends for {dataset_name}")
                plt.show()

        except Exception as e:
            print(f"Error testing on {dataset_name}: {e}")


# Run the Training and Testing
trained_model, test_datasets = train_model()
test_model(trained_model, test_datasets)