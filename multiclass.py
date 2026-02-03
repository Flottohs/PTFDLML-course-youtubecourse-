import torch as torch #hehehehe
import numpy as np


import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

from sklearn.model_selection import train_test_split

NUM_CLASSES = 4
NUM_FEATURES = 2
RANDOM_SEED = 42

X_blob,y_blob = make_blobs(n_samples= 1000,n_features = NUM_FEATURES, centers = NUM_CLASSES, cluster_std= 1.5, random_state = RANDOM_SEED)



X_blob = torch.from_numpy(X_blob).type(torch.float)
y_blob = torch.from_numpy(y_blob).type(torch.long)




X_blob_train, X_blob_test, y_blob_train, y_blob_test = train_test_split(X_blob,y_blob, test_size = 0.2, random_state = RANDOM_SEED)

plt.figure(figsize = (10,7))
plt.scatter(X_blob[:,0], X_blob[:,1],c = y_blob, cmap = plt.cm.RdYlBu)





#creating multi class classification model in pytorch




device = 'cuda' if torch.cuda.is_available() else 'cpu'
from torch import nn

class Blobbingitmodel(nn.Module):
    def __init__(self,input_features,output_features,hidden_units = 8):
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features = input_features, out_features = hidden_units),
            nn.ReLU(),
            nn.Linear(in_features = hidden_units, out_features = hidden_units),
            nn.ReLU(),
            nn.Linear(in_features = hidden_units, out_features = output_features))
        
        
        
    def forward(self,x):
        return self.linear_layer_stack(x)
    
model_4 = Blobbingitmodel(input_features=2, output_features = 4,hidden_units = 128).to(device)


print(model_4)







loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params = model_4.parameters(), lr = 0.01)



#either could create the training loop or predictions


model_4.eval()
with torch.inference_mode():
    y_logits = model_4(X_blob_test.to(device))




#convert logits to pred probs
y_pred_probs = torch.softmax(y_logits,dim = 1)
print(y_logits[:5])

print(y_pred_probs[:5])



pred_labels = torch.argmax(y_pred_probs,dim = 1)
print(pred_labels)



###logits -> pred probs(softmax)-> pred labels(argmax dim =1)


torch.manual_seed(42)

torch.cuda.manual_seed(42)



X_blob_train = X_blob_train.to(device)
X_blob_test = X_blob_test.to(device)

# 2. Move Labels (y) to device and convert to LONG (required for CrossEntropy)
y_blob_train = y_blob_train.to(device)
y_blob_test = y_blob_test.to(device)





def accuracy_fn(y_true, y_pred):
    """Calculates accuracy percentage."""
    correct = torch.eq(y_true, y_pred).sum().item()
    return (correct / len(y_pred)) * 100


epochs = 10000
for epoch in range(epochs):
    model_4.train()
    y_logits = model_4(X_blob_train)
    y_pred = torch.softmax(y_logits,dim = 1).argmax(dim = 1)
    loss = loss_fn(y_logits,y_blob_train)
    acc = accuracy_fn(y_true = y_blob_train,y_pred = y_pred)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    
    model_4.eval()
    with torch.inference_mode():
        test_logits = model_4(X_blob_test)
        test_preds = torch.softmax(test_logits, dim=1).argmax(dim=1) # These are class labels
        test_loss = loss_fn(test_logits,y_blob_test)
        test_acc = accuracy_fn(y_true = y_blob_test, y_pred = test_preds)
        
    if epoch % 100 == 0:
        print(f'epoch: {epoch}| train loss:  {loss:.5f}, train accuracy :{acc:.5f}| test loss: {test_loss:.5}, test accuracy{test_acc:.5f}')
        



#making and evaluating predictions


model_4.eval()
with torch.inference_mode():
    y_logits = model_4(X_blob_test)


y_pred_probs = torch.softmax(y_logits,dim = 1)

#labels
y_preds = torch.argmax(y_pred_probs,dim = 1)

print(y_preds[:10])
print(y_blob_test[:10])

from helper_functions import plot_decision_boundary
plt.figure(figsize = (12,6))
plt.subplot(1,2,1)
plt.title('Train')
plot_decision_boundary(model_4,X_blob_train,y_blob_train)
plt.show()
plt.title('Test')
plot_decision_boundary(model_4,X_blob_test,y_blob_test)
plt.show()


#a few more classification metrics(to evaluate our classification model)

#accuracy
#precision
#recall
#f1 scoree
#confusion matrix
#classification report






from torchmetrics import Accuracy

torchmetric_accuracy = Accuracy(
    task="multiclass",
    num_classes=4
).to(device)

acc = torchmetric_accuracy(y_preds, y_blob_test)
print(acc)
