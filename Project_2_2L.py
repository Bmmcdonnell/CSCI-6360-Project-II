import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from get_qof import get_qof
from latex_tables import is_oos_comparison

# Define the Model
class NoHiddenLayerNN(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.flatten = nn.Flatten() 
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.ReLU()
            # nn.LeakyReLU() 
        )
        self.loss_fn = nn.MSELoss()
        self.lr = 0.01
        self.maxepochs = 400
        self.batch_size = 32
        self.patience = 25      # How many epochs to wait before stopping
        self.min_delta = 1e-4   # Minimum improvement required to reset the patience counter

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

    def trainNN(self, X, y):
        # Set Loss Function and Optimizer
        loss_fn = self.loss_fn
        optimizer = optim.SGD(self.parameters(), lr=self.lr)

        batch_size = self.batch_size

        # Combine X and y into a single dataset, then load it into a DataLoader
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True) 

        epochs = self.maxepochs 
        print(f"Starting training with batch size {batch_size}...")

        # --- EARLY STOPPING TRACKERS ---
        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            epoch_loss = 0.0 
            
            # --- THE MINI-BATCH LOOP ---
            for batch_X, batch_y in dataloader:
                predictions = self(batch_X)
                loss = loss_fn(predictions, batch_y)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() 
            
            # Calculate the average loss across all batches
            avg_loss = epoch_loss / len(dataloader) 

            # Print the progress every 10 epochs
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Avg Loss: {avg_loss:.4f}")

            # --- EARLY STOPPING LOGIC ---
            # Check if the loss improved by at least min_delta
            if best_loss - avg_loss > self.min_delta:
                best_loss = avg_loss
                patience_counter = 0  # Reset counter if we see improvement
            else:
                patience_counter += 1 # Increment counter if no significant improvement

            # Stop training if we've run out of patience
            if patience_counter >= self.patience:
                print(f"\nEarly stopping triggered at Epoch {epoch+1}!")
                print(f"Loss hasn't improved by more than {self.min_delta} for {self.patience} consecutive epochs.")
                break

        print("\nTraining complete!")
    
    def testNN(self, X, y):
        # Set the model to evaluation mode
        self.eval() 
        
        # Need the same loss function to compare against training
        loss_fn = self.loss_fn 
        
        # Disable gradient calculation
        with torch.no_grad(): 
            
            # Make predictions
            predictions = self(X) 
            
            # Calculate the test loss
            test_loss = loss_fn(predictions, y)
            
        print(f"Test Loss: {test_loss.item():.4f}")
        
        # Set the model back to training mode just in case
        self.train() 
        
        return (predictions, test_loss.item())

def p2_auto_mpg_2L():
    # ==========================================
    # --- Data Loading ---
    # ==========================================
    oxy = pd.read_csv("datasets/cleaned_auto_mpg_with_intercept.csv")
    ox = oxy.drop('mpg', axis=1)
    X = ox.drop('intercept', axis=1)
    y = oxy[['mpg']]

    # ==========================================
    # --- Train-Test Split (80-20) ---
    # ==========================================
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    scalerIS = StandardScaler()
    X_scaled = scalerIS.fit_transform(X)

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y.to_numpy(), dtype=torch.float32)

    scalerOOS = StandardScaler()
    X_train_scaled = scalerOOS.fit_transform(X_train)
    X_test_scaled = scalerOOS.transform(X_test)

    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.to_numpy(), dtype=torch.float32)
    
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.to_numpy(), dtype=torch.float32)

    # Setup Variables and Instantiate Model
    input_features = X.shape[1]
    output_classes = y.shape[1]
    model = NoHiddenLayerNN(input_size=input_features, output_size=output_classes)

    model.trainNN(X_tensor,y_tensor)
    (IS_predictions, IS_loss) = model.testNN(X_tensor, y_tensor)

    y_numpy = y.to_numpy()
    preds_IS_numpy = IS_predictions.cpu().numpy()
    k = X.shape[1]
    qof_IS = get_qof(y_numpy, preds_IS_numpy, k)

    # print(qof_IS)

    model.trainNN(X_train_tensor,y_train_tensor)
    (OOS_predictions, OOS_loss) = model.testNN(X_test_tensor,y_test_tensor)

    y_test_numpy = y_test.to_numpy()  
    preds_OOS_numpy = OOS_predictions.cpu().numpy()
    k = X_train_scaled.shape[1]
    qof_OOS = get_qof(y_test_numpy, preds_OOS_numpy, k)

    # print(qof_OOS)

    is_oos_comparison(qof_IS, qof_OOS, "Auto MPG", "2LNN")

def p2_housing_2L():
    # ==========================================
    # --- Data Loading ---
    # ==========================================
    oxy = pd.read_csv("datasets/cleaned_housing_with_intercept.csv")
    ox = oxy.drop('median_house_value', axis=1)
    X = ox.drop('intercept', axis=1)
    y = oxy[['median_house_value']]

    # ==========================================
    # --- Train-Test Split (80-20) ---
    # ==========================================
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    scalerIS = StandardScaler()
    X_scaled = scalerIS.fit_transform(X)

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y.to_numpy(), dtype=torch.float32)

    scalerOOS = StandardScaler()
    X_train_scaled = scalerOOS.fit_transform(X_train)
    X_test_scaled = scalerOOS.transform(X_test)

    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.to_numpy(), dtype=torch.float32)
    
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.to_numpy(), dtype=torch.float32)

    # Setup Variables and Instantiate Model
    input_features = X.shape[1]
    output_classes = y.shape[1]
    model = NoHiddenLayerNN(input_size=input_features, output_size=output_classes)

    model.trainNN(X_tensor,y_tensor)
    (IS_predictions, IS_loss) = model.testNN(X_tensor, y_tensor)

    y_numpy = y.to_numpy()
    preds_IS_numpy = IS_predictions.cpu().numpy()
    k = X.shape[1]
    qof_IS = get_qof(y_numpy, preds_IS_numpy, k)

    # print(qof_IS)

    model.trainNN(X_train_tensor,y_train_tensor)
    (OOS_predictions, OOS_loss) = model.testNN(X_test_tensor,y_test_tensor)

    y_test_numpy = y_test.to_numpy()  
    preds_OOS_numpy = OOS_predictions.cpu().numpy()
    k = X_train_scaled.shape[1]
    qof_OOS = get_qof(y_test_numpy, preds_OOS_numpy, k)

    # print(qof_OOS)

    is_oos_comparison(qof_IS, qof_OOS, "California House Prices", "2LNN")



def p2_insurance_2L():
    # ==========================================
    # --- Data Loading ---
    # ==========================================
    oxy = pd.read_csv("datasets/cleaned_insurance_with_intercept.csv")
    ox = oxy.drop('charges', axis=1)
    X = ox.drop('intercept', axis=1)
    y = oxy[['charges']]

    # ==========================================
    # --- Train-Test Split (80-20) ---
    # ==========================================
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    scalerIS = StandardScaler()
    X_scaled = scalerIS.fit_transform(X)

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y.to_numpy(), dtype=torch.float32)

    scalerOOS = StandardScaler()
    X_train_scaled = scalerOOS.fit_transform(X_train)
    X_test_scaled = scalerOOS.transform(X_test)

    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.to_numpy(), dtype=torch.float32)
    
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.to_numpy(), dtype=torch.float32)

    # Setup Variables and Instantiate Model
    input_features = X.shape[1]
    output_classes = y.shape[1]
    model = NoHiddenLayerNN(input_size=input_features, output_size=output_classes)

    model.trainNN(X_tensor,y_tensor)
    (IS_predictions, IS_loss) = model.testNN(X_tensor, y_tensor)

    y_numpy = y.to_numpy()
    preds_IS_numpy = IS_predictions.cpu().numpy()
    k = X.shape[1]
    qof_IS = get_qof(y_numpy, preds_IS_numpy, k)

    # print(qof_IS)

    model.trainNN(X_train_tensor,y_train_tensor)
    (OOS_predictions, OOS_loss) = model.testNN(X_test_tensor,y_test_tensor)

    y_test_numpy = y_test.to_numpy()  
    preds_OOS_numpy = OOS_predictions.cpu().numpy()
    k = X_train_scaled.shape[1]
    qof_OOS = get_qof(y_test_numpy, preds_OOS_numpy, k)

    # print(qof_OOS)

    is_oos_comparison(qof_IS, qof_OOS, "Insurance Charges", "2LNN")


# p2_auto_mpg_2L()
# p2_housing_2L()
# p2_insurance_2L()