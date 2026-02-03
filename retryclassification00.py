import torch
from torch import nn
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split

# 1. Data Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
X, y = make_circles(n_samples=1000, noise=0.03, random_state=42)

# Convert to tensors and split
X = torch.from_numpy(X).type(torch.float).to(device)
y = torch.from_numpy(y).type(torch.float).to(device)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Model Definition (Sequential is much shorter for simple architectures)
model = nn.Sequential(
    nn.Linear(2, 10),

    nn.Linear(10, 10),

    nn.Linear(10, 1)
).to(device)

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

def get_acc(y_true, y_logits):
    preds = torch.round(torch.sigmoid(y_logits))
    return (torch.eq(y_true, preds).sum().item() / len(preds)) * 100

# 3. Training Loop
torch.manual_seed(42)
for epoch in range(10000):
    model.train()
    logits = model(X_train).squeeze()
    loss = loss_fn(logits, y_train)
    acc = get_acc(y_train, logits)

    optimizer.zero_grad()
    loss.backward()
    
    
    
    
    
    
    
    optimizer.step()

    if epoch % 20 == 0:
        model.eval()
        with torch.inference_mode():
            t_logits = model(X_test).squeeze()
            t_loss = loss_fn(t_logits, y_test)
            t_acc = get_acc(y_test, t_logits)
            print(f"Epoch: {epoch} | Loss: {loss:.4f}, Acc: {acc:.2f}% | Test Loss: {t_loss:.4f}, Test Acc: {t_acc:.2f}%")




# 4. Visualization (Requires helper_functions.py)
from helper_functions import plot_decision_boundary
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1); plt.title("Train")
plot_decision_boundary(model, X_train, y_train)
plt.subplot(1, 2, 2); plt.title("Test")
plot_decision_boundary(model, X_test, y_test)
plt.show()


#must figure out if its actually learning
#preparing data to see if our model can fit a straght line
#one way to troub;eshoot a larger problem is to test a smaller problem

weight = 0.7
bias = 0.3
start =0
end = 1
step = 0.01



X_regression= torch.arange(start,end,step).unsqueeze(dim = 1)

y_regression =weight * X_regression +bias 




train_split= int(0.8 * len(X_regression))


X_train_regression ,y_train_regression = X_regression[:train_split],y_regression[:train_split]

X_test_regression, y_test_regression = X_regression[train_split:],y_regression[train_split:]

print(len(X_train_regression),len(y_train_regression),len(X_test_regression),len(y_test_regression))




def plot_predictions(train_data = X_train,
                     train_labels = y_train,
                     test_data = X_test,
                     test_labels = y_test,
                     predictions = None):
    plt.figure(figsize=(10,7))
    plt.scatter(train_data,train_labels,c='b',s=4,label='Training data')
    plt.scatter(test_data,test_labels,c='g',s=4,label='Testing data')
    if predictions is not None:
        plt.scatter(test_data,predictions,c='r',s=4,label='Predictions')
    plt.legend()
    plt.show()

plot_predictions(train_data = X_train_regression,train_labels = y_train_regression,test_data = X_test_regression,test_labels = y_test_regression)


model_2 = nn.Sequential(
    nn.Linear(in_features = 1, out_features=10),
    nn.Linear(in_features=10, out_features =10),
    nn.Linear(in_features = 10, out_features = 1)
)

print(model_2)

loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params =model_2.parameters(),lr = 0.1)

torch.manual_seed(42)
epochs = 1000

model_2.to(device)
X_train_regression,y_train_regression,X_test_regression,y_test_regression = X_test_regression.to(device),y_train_regression.to(device),X_test_regression.to(device),y_test_regression.to(device)


#trianing
for epoch in range(epochs):
    y_pred = model_2(X_train_regression)
    loss = loss_fn(y_pred,y_train_regression)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    
    
    
    #test
    model_2.eval()
    with torch.inference_mode():
        test_pred = model_2(X_test_regression)
        testloss = loss_fn(test_pred,y_test_regression) 
    if epoch %100 == 0:
        print(f'epoch: {epoch}| loss: {loss:.5f}|testloss:{testloss:.5f}')
        
        


model_2.eval()

#make predictions(inference)
with torch.inferencemode():
    y_preds = model_2(X_test_regression)
    #plot data and predicitions
    
    
plot_predictions(train_data = X_train_regression.cpu(), train_labels = y_train_regression.cpu(), test_data= X_test_regression.cpu(), test_labels = y_test_regression.cpu())






