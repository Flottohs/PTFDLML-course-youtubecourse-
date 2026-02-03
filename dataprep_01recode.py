import torch
from torch import nn 
from pathlib import Path
import matplotlib.pyplot as plt




#check pytorch version:
print(torch.__version__)
'2.9.0'



#create device agnostic code

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(device)




#dummy data set using linear regression formula as y = mx + c
weight = .7
bias = 0.3

#create range values

start = 0
end = 1
step = 0.02



X = torch.arange(start,end,step).unsqueeze(dim = 1)

y= weight * X + bias
#training_split
train_split = int(.8 * len(X))
X_train,y_train = X[:train_split],y[:train_split]
X_test,y_test = X[train_split:],y[train_split:]

X_train = X_train.to(device)
y_train = y_train.to(device)
X_test= X_test.to(device)
y_test = y_test.to(device)

print(len(X_test),len(X_train),len(y_test),len(y_train))


#plot data




def plot_predictions(train_data = X_train,
                     train_labels = y_train,
                     test_data = X_test,
                     test_labels = y_test,
                     predictions = None):
    plt.figure(figsize=(10,7))
    plt.scatter(train_data.cpu(), train_labels.cpu(), c='b', s=4, label='Training data')
    plt.scatter(test_data.cpu(), test_labels.cpu(), c='g', s=4, label='Testing data')
    if predictions is not None:
        plt.scatter(test_data.cpu(), predictions.cpu(), c='r', s=4, label='Predictions')
    plt.legend()
    plt.show(block = True)



#building pytorch linear model



#create linear model by subclassing nn.module

class LinearRegressionModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        #use linear for creating parameters
        self.linear_layer = nn.Linear(in_features = 1,out_features= 1 )# for every x value output one y value, 1to1 mapping
        
        
    def forward(self,x:torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)
#manual seed
torch.manual_seed(42)
model_1 = LinearRegressionModelV2()
print(model_1.state_dict())




#set the model to use the target device


model_1.to(device)




##training
#loss function
#optimizer
#training and testing loop

loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params = model_1.parameters(),lr=0.01)


#training loop
torch.manual_seed(42)

plot_predictions(X_train, y_train, X_test, y_test)

epochs = 200

for epoch in range(epochs):
    model_1.train()
    
    y_pred = model_1(X_train)
    
    
    loss = loss_fn(y_pred,y_train)
    
    optimizer.zero_grad()
    loss.backward()
    
    optimizer.step()
    
    model_1.eval()
    
    with torch.inference_mode():
        test_pred = model_1(X_test)
        test_loss = loss_fn(test_pred,y_test)
        
    if epoch %10 == 0:
        print(model_1.state_dict())
        
    
    
model_1.eval()

with torch.inference_mode():
    y_preds = model_1(X_test)
print(y_preds)


plot_predictions(predictions = y_preds)




from pathlib import Path
model_path = Path('models')

model_name = '01_pytorch_workflow_model_1.pth'

model_save_path = model_path/model_name

torch.save(obj = model_1.state_dict(), f = model_save_path)


loaded_model_1 = LinearRegressionModelV2()

loaded_model_1.load_state_dict(torch.load(f = model_save_path))



#put target model must use device agnostic code

loaded_model_1.to(device)


print(next(loaded_model_1.parameters()).device)




loaded_model_1.eval()

with torch.inference_mode():
    loaded_model_1_preds= loaded_model_1(X_test)
print(y_preds ==loaded_model_1_preds)