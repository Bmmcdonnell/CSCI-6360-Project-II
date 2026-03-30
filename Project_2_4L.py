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
from save_plots import save_sorted_plot

class TwoHiddenLayerNNLeakyRelu(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(TwoHiddenLayerNNLeakyRelu, self).__init__()
        
        # Layers
        self.fc1 = nn.Linear(input_size, hidden_size1)   # Input → Hidden 1
        self.fc2 = nn.Linear(hidden_size1, hidden_size2) # Hidden 1 → Hidden 2
        self.fc3 = nn.Linear(hidden_size2, output_size)  # Hidden 2 → Output
        
        # Activation
        self.leaky_relu = nn.LeakyReLU()
        
        # Training params
        self.batch_size = 32
        self.patience = 25
        self.min_delta = 1e-4
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.leaky_relu(out)
        
        out = self.fc2(out)
        out = self.leaky_relu(out)
        
        out = self.fc3(out)
        return out

    def trainLeakyRelu(self, X_train, y_train, max_epochs=200, learning_rate=0.01):
        self.train()
        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)

        dataset = TensorDataset(X_train, y_train)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(max_epochs):
            epoch_loss = 0.0
            
            for batch_X, batch_y in dataloader:
                outputs = self(batch_X)
                loss = criterion(outputs, batch_y)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{max_epochs} | Avg Loss: {avg_loss:.4f}")

            if best_loss - avg_loss > self.min_delta:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                print(f"\nEarly stopping triggered at Epoch {epoch+1}!")
                print(f"Loss hasn't improved by more than {self.min_delta} for {self.patience} consecutive epochs.")
                break
            
        print("Training Complete!")

    def testLeakyRelu(self, X_test, y_test):
        self.eval()
        loss_fn = nn.MSELoss()

        with torch.no_grad():
            outputs = self(X_test)
            test_loss = loss_fn(outputs, y_test)

        print(f"Test Loss: {test_loss.item():.4f}")
        self.train()
        return (outputs, test_loss.item())
    

def p2_auto_mpg_4L():
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
    # input_features = X.shape[1]
    # output_classes = y.shape[1]
    model = TwoHiddenLayerNNLeakyRelu(input_size=X.shape[1], hidden_size1=100, hidden_size2=50, output_size=y.shape[1])

    model.trainLeakyRelu(X_tensor,y_tensor)
    (IS_predictions, IS_loss) = model.testLeakyRelu(X_tensor, y_tensor)

    y_numpy = y.to_numpy()
    preds_IS_numpy = IS_predictions.cpu().numpy()
    k = X.shape[1]
    qof_IS = get_qof(y_numpy, preds_IS_numpy, k)

    save_sorted_plot(np.ravel(y_numpy), np.ravel(preds_IS_numpy), "Auto MPG", "Auto_MPG_P2", "4L NN", "4L", False)

    # print(qof_IS)

    model.trainLeakyRelu(X_train_tensor,y_train_tensor)
    (OOS_predictions, OOS_loss) = model.testLeakyRelu(X_test_tensor,y_test_tensor)

    y_test_numpy = y_test.to_numpy()  
    preds_OOS_numpy = OOS_predictions.cpu().numpy()
    k = X_train_scaled.shape[1]
    qof_OOS = get_qof(y_test_numpy, preds_OOS_numpy, k)

    save_sorted_plot(np.ravel(y_test_numpy), np.ravel(preds_OOS_numpy), "Auto MPG", "Auto_MPG_P2", "4L NN", "4L", True)

    # print(qof_OOS)

    is_oos_comparison(qof_IS, qof_OOS, "Auto MPG", "4L NN")

def p2_housing_4L():
    # ==========================================
    # --- Data Loading ---
    # ==========================================
    oxy = pd.read_csv("datasets/cleaned_housing_with_intercept.csv")
    ox = oxy.drop('median_house_value', axis=1)
    X = ox.drop('intercept', axis=1)
    y = oxy[['median_house_value']] / 100000.0

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
    # input_features = X.shape[1]
    # output_classes = y.shape[1]
    model = TwoHiddenLayerNNLeakyRelu(input_size=X.shape[1], hidden_size1=100, hidden_size2=50, output_size=y.shape[1])

    model.trainLeakyRelu(X_tensor,y_tensor)
    (IS_predictions, IS_loss) = model.testLeakyRelu(X_tensor, y_tensor)

    y_numpy = y.to_numpy() * 100000.0
    preds_IS_numpy = IS_predictions.cpu().numpy() * 100000.0
    k = X.shape[1]
    qof_IS = get_qof(y_numpy, preds_IS_numpy, k)

    save_sorted_plot(np.ravel(y_numpy), np.ravel(preds_IS_numpy), "California House Prices", "Housing_P2", "4L NN", "4L", False)

    # print(qof_IS)

    model.trainLeakyRelu(X_train_tensor,y_train_tensor)
    (OOS_predictions, OOS_loss) = model.testLeakyRelu(X_test_tensor,y_test_tensor)

    y_test_numpy = y_test.to_numpy() * 100000.0
    preds_OOS_numpy = OOS_predictions.cpu().numpy() * 100000.0
    k = X_train_scaled.shape[1]
    qof_OOS = get_qof(y_test_numpy, preds_OOS_numpy, k)

    save_sorted_plot(np.ravel(y_test_numpy), np.ravel(preds_OOS_numpy), "California House Prices", "Housing_P2", "4L NN", "4L", True)

    # print(qof_OOS)

    is_oos_comparison(qof_IS, qof_OOS, "California House Prices", "4L NN")



def p2_insurance_4L():
    # ==========================================
    # --- Data Loading ---
    # ==========================================
    oxy = pd.read_csv("datasets/cleaned_insurance_with_intercept.csv")
    ox = oxy.drop('charges', axis=1)
    X = ox.drop('intercept', axis=1)
    y = oxy[['charges']] / 6000.0

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
    # input_features = X.shape[1]
    # output_classes = y.shape[1]
    model = TwoHiddenLayerNNLeakyRelu(input_size=X.shape[1], hidden_size1=100, hidden_size2=50, output_size=y.shape[1])

    model.trainLeakyRelu(X_tensor,y_tensor)
    (IS_predictions, IS_loss) = model.testLeakyRelu(X_tensor, y_tensor)

    y_numpy = y.to_numpy() * 6000.0
    preds_IS_numpy = IS_predictions.cpu().numpy() * 6000.0
    k = X.shape[1]
    qof_IS = get_qof(y_numpy, preds_IS_numpy, k)

    save_sorted_plot(np.ravel(y_numpy), np.ravel(preds_IS_numpy), "Insurance Charges", "Insurance_P2", "4L NN", "4L", False)

    # print(qof_IS)

    model.trainLeakyRelu(X_train_tensor,y_train_tensor)
    (OOS_predictions, OOS_loss) = model.testLeakyRelu(X_test_tensor,y_test_tensor)

    y_test_numpy = y_test.to_numpy() * 6000.0
    preds_OOS_numpy = OOS_predictions.cpu().numpy() * 6000.0
    k = X_train_scaled.shape[1]
    qof_OOS = get_qof(y_test_numpy, preds_OOS_numpy, k)

    save_sorted_plot(np.ravel(y_test_numpy), np.ravel(preds_OOS_numpy), "Insurance Charges", "Insurance_P2", "4L NN", "4L", True)

    # print(qof_OOS)

    is_oos_comparison(qof_IS, qof_OOS, "Insurance Charges", "4L NN")


# p2_auto_mpg_4L()
# p2_housing_4L()
# p2_insurance_4L()