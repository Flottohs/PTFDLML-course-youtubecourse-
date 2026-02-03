import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
print(torch.__version__)

print(torch.cuda.is_available())




#data preparing and loading
#2 parts of NN

#get data into a numerical representation
#build model to learn patterns in that numerical representation



#to showcase this lets create some data with the linear regression formula

#y=mx+b


#well use this formuala to create a straight line with known parameters so that the nn can learn from it

#create known parameters
weight = 0.7
bias = 0.3

#create 
start =0
end = 1
step =0.02
X = torch.arange(start,end,step).unsqueeze(dim=1) #features adds extra dimension
Y = weight * X + bias #labels/targets
print(X[:10],Y[:10],len(X),len(Y))





#split data into training and test sets
#training, validation, test sets
#course materials, practice exam ,final exam



#60-80% training,validation 10-20, 10-20% test


#simple train/test split
train_split = int(0.8 * len(X))
X_train, Y_train = X[:train_split], Y[:train_split]
X_test, Y_test = X[train_split:], Y[train_split:]


print(len(X_train),len(Y_train),len(X_test),len(Y_test))


#visualize data

#visualize visualize visualize
def plot_predictions(train_data = X_train,
                     train_labels = Y_train,
                     test_data = X_test,
                     test_labels = Y_test,
                     predictions = None):
    plt.figure(figsize=(10,7))
    plt.scatter(train_data,train_labels,c='b',s=4,label='Training data')
    plt.scatter(test_data,test_labels,c='g',s=4,label='Testing data')
    if predictions is not None:
        plt.scatter(test_data,predictions,c='r',s=4,label='Predictions')
    plt.legend()
    plt.show()
    
plot_predictions()



#create a linear regression model class
#what the model does:
#start with random values
#look at trainign data and adjust the random values and aadjusts them to get closer to the data

#how does it do this:
#gradient descent and backpropagation


class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1,requires_grad=True,dtype = torch.float))
        self.bias = nn.Parameter(torch.randn(1,requires_grad=True,dtype = torch.float))
    #forward method to define computeation in the model
    
                    #type hint: saying x is tensor, -> is also a type hint saying it will retunrn a tensor
    def forward(self,x:torch.Tensor) -> torch.Tensor:#x is input tensor
        return self.weights * x + self.bias






#checking contents of our model

#can check with .parameters


#create random seed
torch.manual_seed(42)

#create instance
model_0 = LinearRegressionModel()
print(model_0)
print(list(model_0.parameters()))

#list named parameters
print(model_0.state_dict())



#making predictions using torch.inference mode
#to check out models power, predict y_test based on x_test
#when we pass data through our model its going to run through the forward method
model_0.eval()
with torch.inference_mode():# disables gradient tracking, speeds up computations, reduces memory usage
    y_preds = model_0(X_test)
print(y_preds)

print(Y_test)


plot_predictions(predictions=y_preds)
#because it starts with random values, the predictions are random and clearly bad








#learning- move unknown parameters to known parameters
#or in other words to move from a bad representation to a good representation


#loss function(cost function)- a function to measure how wrong the model is
#optimization- how we improve the model to reduce the loss(changing weights and biases)


#training loop
#testing loop




#L1loss- mean absolute error
#L2loss- mean squared error


LOSS_FN = nn.L1Loss()#MAE

#optimizer = stochastic gradient descent- random- drunk man walk
optimizer = torch.optim.SGD(params=model_0.parameters(),lr=0.01)#learning rate changese how big of a step we take to reach the optimal value





#building a training loop and a testing loop
#training loop
#forward pass also called a forward propagation
#calculate the loss(compare forward pass predictions to ground truth labels_
#optimizer zero grad
#loss backward= move backwards through the network to calculate the gradients of each of the gradients of the paramets of our model with respect to the loss
#optimizer step

#epoch is one loop throhgh all the data
epochs = 1000
epoch_count = []
loss_values = []
test_loss_values = []


for epoch in range(epochs):
    model_0.train()#put model in training mode,
    y_preds = model_0(X_train)#forward pass
    loss = LOSS_FN(y_preds,Y_train)#calculate loss
    optimizer.zero_grad()#zero gradients
    loss.backward()#backpropagation
    optimizer.step()#update parameters
    
    #testing loop
    model_0.eval()#put model in evaluation mode,turns of gradient tracking
    with torch.inference_mode():
        test_preds = model_0(X_test)
        test_loss = LOSS_FN(test_preds,Y_test)
        
    if epoch % 10 ==0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f} | Test loss: {test_loss:.5f}")
        epoch_count.append(epoch)
        loss_values.append(loss)
        test_loss_values.append(test_loss)
        



print(model_0.state_dict())



with torch.inference_mode():
    y_preds_new= model_0(X_test)
    
plot_predictions(predictions = y_preds_new)
plt.plot(epoch_count,np.array(torch.tensor(loss_values).cpu().numpy()),label ='train_loss' )
plt.plot(epoch_count,test_loss_values, label = 'test loss')
plt.title('training and test loss curves')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend()



#different ways to save and load your models:
#1: torch.save() - alwasys you to save your object in pythons pickle form
#2torch.load() alows you to load your pytorch object
#torch.nn.Module.load_state_dict() allows to load a models saved state_dict


#A state_dict is simply a Python dictionary object that maps each layer to its parameter tensor.



#saving the model

from pathlib import Path 

#create directory
MODEL_PATH = Path('/Users/haydenfletcher/Documents/programming/books-course/PTFDLML/models')


MODEL_NAME = '00_pytorch_workflow_model_0.pth'
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME


print(f"I am looking for the file here: {MODEL_SAVE_PATH.resolve()}")
print(f"Does the folder actually exist? {MODEL_PATH.exists()}")

print(f'saving model to {MODEL_SAVE_PATH}')
torch.save(obj = model_0.state_dict(),f = MODEL_SAVE_PATH)



#since we saved the models state_dict, rather than the entire model, we will create a new nstance of our model class and load the saved state_dict() into that

#to load in a saved state_dict we have to instantiate a new instance of our model class
print(model_0.state_dict())
loaded_model_0 = LinearRegressionModel()
#load the saved state_dict of model0(this will update the new isntance with the updated parameters)
print(loaded_model_0.state_dict())
loaded_model_0.load_state_dict(torch.load(f = MODEL_SAVE_PATH))
print(loaded_model_0.state_dict())



#just to make sure
model_0.eval()
with torch.inference_mode():
    y_preds = model_0(X_test)

loaded_model_0.eval()
with torch.inference_mode():
    loaded_model_preds = loaded_model_0(X_test)
    

print(loaded_model_preds == y_preds)




