import matplotlib.pyplot as plt
from sklearn.datasets import make_circles

import torch as torch
device = "cuda" if torch.cuda.is_available() else "cpu"




n_samples = 1000

X,y = make_circles(n_samples, noise = 0.03, random_state = 42
                   )



plt.scatter(X[:,0],X[:,1],c=y,cmap = plt.cm.RdYlBu)
plt.show()

import torch
from sklearn.model_selection import train_test_split
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)


#split


X_train,X_test,y_train,y_test = train_test_split(X,y,train_size = 0.8,random_state = 42)


print(X_train[:5],y_train[:5])


#bulding odel with non linearity


from torch import nn


class CircleModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        #layers
        self.layer_1 = nn.Linear(in_features = 2,out_features= 10)
        self.layer_2 = nn.Linear(in_features = 10,out_features =10)
        self.layer_3 = nn.Linear(in_features = 10, out_features = 1)
        self.ReLU = nn.ReLU()
        
        
    def forward(self,x):
        return(self.layer_3(self.ReLU(self.layer_2(self.ReLU(self.layer_1(x))))))
    
    
    
model_3 = CircleModelV2().to(device)


loss_FN = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model_3.parameters(),lr = 0.03)

def get_acc(y_true, y_logits):
    preds = torch.round(torch.sigmoid(y_logits))
    return (torch.eq(y_true, preds).sum().item() / len(preds)) * 100


X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)




epochs = 10000
for epochs in range(epochs):
    model_3.train()
    
    #forward pass
    y_logits = model_3(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))
    loss = loss_FN(y_logits,y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    acc = get_acc(y_train, y_logits)
    
    
    #ttest loop
    model_3.eval()
    with torch.inference_mode():
        y_test_logits = model_3(X_test).squeeze()
        y_pred_test = torch.round(torch.sigmoid(y_test_logits))
        test_loss = loss_FN(y_test_logits, y_test)  # compare logits to true labels
        t_acc = get_acc(y_test, y_test_logits)
    if epochs % 100 == 0:
        print(f'epoch: {epochs}|train loss: {loss:.5f}, test accuracy:{acc}| test loss: {test_loss:.5f},test accuracy: {t_acc}')
        
    
    
    
    
    
    

    
    
#make predictions
model_3.eval()
with torch.inference_mode():
    y_preds = torch.sigmoid(model_3(X_test).squeeze())

#plot


import numpy as np

def plot_decision_boundary(model, X, y):
    # Move model and data to CPU for numpy/matplotlib compatibility
    model.to("cpu")
    X, y = X.to("cpu"), y.to("cpu")

    # Setup meshgrid for plotting
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101),
                         np.linspace(y_min, y_max, 101))

    # Prepare features for prediction
    X_to_pred_on = torch.from_numpy(np.column_stack((xx.ravel(), yy.ravel()))).float()

    # Make predictions
    model.eval()
    with torch.inference_mode():
        y_logits = model(X_to_pred_on)

    # Adjust logits to predicted labels
    if len(torch.unique(y)) > 2:
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1) # multi-class
    else:
        y_pred = torch.round(torch.sigmoid(y_logits)) # binary

    # Reshape predictions and plot
    y_pred = y_pred.reshape(xx.shape).detach().numpy()
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("Model Decision Boundary")
    plt.show()

# Run the plotting function
plot_decision_boundary(model_3, X_test, y_test)






##replicating non linear activation functions


#linear and non linear functions
#create a tensor
A = torch.arange(-10,10,1,dtype  = torch.float32)
print(A.dtype)

#VISUALIZE VISUALIZE VISUALIZE RAHHH
plt.plot(A)
plt.show()
plt.close()
plt.plot(torch.relu(A))


def relu(x:torch.Tensor) -> torch.Tensor:
    return torch.maximum(torch.tensor(x),torch.tensor(0))

plt.plot(relu(A))


#now for sigmoid

def sigmoid(x):
    return 1/(1 + torch.exp(-x))


plt.plot(sigmoid(A))


