import pandas as pd
import torch as ts 
import torch.nn as nn

# reading file
df = pd.read_csv("powerplant_data.csv")

# sorting out features and label
X = df.drop(["PE"] , axis = 1)
y = df["PE"]

# data split
from sklearn.model_selection import train_test_split
X_train , X_test  , y_train , y_test = train_test_split(X , y , test_size=0.2 , random_state= 42)

# standardization
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()

X_train_scaled = ss.fit_transform(X_train)
X_test_scaled = ss.transform(X_test)

X_train_tensor = ts.tensor(X_train_scaled, dtype= ts.float32)
X_test_tensor = ts.tensor(X_test_scaled, dtype= ts.float32)
y_train_tensor = ts.tensor(y_train.values, dtype= ts.float32).view(-1 , 1)
y_test_tensor = ts.tensor(y_test.values , dtype= ts.float32 ).view(-1 , 1)

# dataset and dataloader 

from torch.utils.data import TensorDataset , DataLoader

train_dataset = TensorDataset(X_train_tensor ,y_train_tensor )
test_dataset = TensorDataset(X_test_tensor , y_test_tensor)

train_loader = DataLoader(train_dataset , batch_size = 32 , shuffle= True)
test_loader = DataLoader(test_dataset , batch_size = 32 )


#model
class ann_model(nn.Module):

    def __init__(self):
        super(ann_model , self).__init__()

        self.model= nn.Sequential(

            # 1st hiden layer 
            nn.Linear(X_train.shape[1] , 6),
            nn.ReLU(),

            # 2nd hidden layer 
            nn.Linear(6,6),
            nn.ReLU(),

            #output
            nn.Linear(6,1)    
        )
    def forward(self , X):
        return self.model(X)

model = ann_model()

cartirian = nn.MSELoss()
opt =  ts.optim.Adam(model.parameters())

epochs = 100
train_losses = []
eval_losses = []
best_model_para = float("inf")

for epoch in range(epochs):
    
# TRAINING THE MODEL
    
    model.train()
    runninng_loss = 0.00
    for xa , yb in train_loader:
        
        opt.zero_grad()
        outputs = model(xa)
        loss = cartirian(outputs , yb)
        loss.backward()
        opt.step()
        runninng_loss += loss.item()
    epoch_training_losses = (runninng_loss/len(train_loader))
    train_losses.append(epoch_training_losses)
    
# Validation
    model.eval()
    running_vaL_loss = 0.00
    with ts.no_grad():
         
         for xa , yb in test_loader:
             
             outputs = model(xa)
             loss = cartirian( outputs , yb)
             running_vaL_loss += loss.item()
    epoch_val_losses = (running_vaL_loss/len(test_loader))
    eval_losses.append(epoch_val_losses)
    
    if epoch_val_losses < best_model_para :
        best_model_para = epoch_val_losses
        ts.save(model.state_dict(), "best_model.pt")
    print(f"epoch {epoch} losses ===> training loss {epoch_training_losses} $ val_losses = {epoch_val_losses}")

# plotting 
import matplotlib.pyplot as plt

loss_df = pd.DataFrame({
    "Training Loss": train_losses,
    "Validation Loss": eval_losses
})

plt.plot(loss_df["Training Loss"], label = "Training Loss")
plt.plot(loss_df["Validation Loss"], label = "Validation Loss")

plt.xlabel("Epochs")
plt.ylabel("Losses")

plt.legend()
plt.show()
             
# loading the best model
model.load_state_dict(ts.load("best_model.pt"))

# evalution
model.eval()
with ts.no_grad():
    train_pred = model(X_train_tensor)
    test_pred = model(X_test_tensor)

    train_mse = cartirian(train_pred , y_train_tensor)
    test_mse = cartirian(test_pred , y_test_tensor)

from sklearn.metrics import r2_score

r2 = r2_score(y_test , test_pred)

print(f"trainMSE = {train_mse} test MSE = {test_mse} and R2 = {r2}")
