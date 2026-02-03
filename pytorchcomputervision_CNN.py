import os
import torch
from torch import nn
import torchvision as tv
import torchvision.datasets
import torchvision.models
import torchvision.transforms
import torch.utils.data.dataset
import torch.utils.data.dataloader
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt



device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(torch.__version__)

print(torchvision.__version__)


#getting a dataset
from pathlib import Path



from torchvision import datasets



train_data = datasets.FashionMNIST(#gets training dataset off fashion.mnist
    root="data",#specifies the root directory where the dataset is stored
    train=True, #specifies training dataset
    download=True,
    transform=ToTensor(),#converts the images to PyTorch tensors
    target_transform= None#how do we want to transform the labels
    )

test_data = datasets.FashionMNIST(#gets test data fro fashion.mnist
    root="data",
    train=False,#specifies test dataset
    download=True,#downloads the data from the internet if it’s not available at root.
    transform=ToTensor(),
    target_transform= None #how do we want to transform the labels
)



print(len(train_data),len(test_data))



image, label = train_data[0]
testimage, testlabel = test_data[0]
class_names = train_data.classes
print(f'the different classes{class_names}')


class_to_idx = train_data.class_to_idx
print(f'the different classes and there index{class_to_idx}')

print(f'training data targets{train_data.targets}')
print(f'image shape{image.shape} ->[colour_channels,heigh,width],')
#28/28


#the colour channel is only one because its gray scale meaning that its only black and white
print(label)

#matplot lib doesnt expect the extra colour dimensions so we have to get rid of it useing squeeze

#matplot lib interprets the colours as not gray, so we have to specify gray scale
'''plt.imshow(image.squeeze(), cmap = 'gray')
plt.title(class_names[label])
plt.show()
'''

#plot more images

torch.manual_seed(42)
'''fig  = plt.figure(figsize = (9,9))'''
rows, cols =4,4
'''
for i in range(1, rows*cols +1):
    random_idx = torch.randint(0,len(train_data),size = [1]).item()
    img,label = train_data[random_idx]
    fig.add_subplot(rows,cols,i)
    plt.imshow(img.squeeze(),cmap = 'gray')
    plt.title(class_names[label])
    
    plt.axis(False)
    
    
plt.show()
'''
##prepare dataloader

#currently the data is in datasets, need to convert to a dataloader

#dataloader turns our dataset into a python iterable

#and we need to batch it


#this is needed as its more computationaly efficient and it allows the nn to adjust the gradients many more times



BATCH_SIZE = 32

from torch.utils.data import DataLoader


train_dataloader = DataLoader(dataset = train_data, batch_size = BATCH_SIZE, shuffle = True)



test_dataloader = DataLoader(dataset = test_data,batch_size = BATCH_SIZE,  shuffle = False)


print(f'dataloaders :{train_dataloader,test_dataloader}')

print(f'length of training data loader{len(train_dataloader)} with batches of {BATCH_SIZE}')
print(f'length of training data loader{len(test_dataloader)} with batches of {BATCH_SIZE}')

train_features_batch,train_labels_batch = next(iter(train_dataloader))
print(train_features_batch.shape)
print(train_labels_batch.shape)



#VISUALIZE VISUALIZE VISUALIZE BOIIII


torch.manual_seed(42)
random_idx = torch.randint(0,len(train_features_batch),size = [1]).item()

img,label = train_features_batch[random_idx], train_labels_batch[random_idx]

'''

plt.imshow(img.squeeze(), cmap = 'gray')
plt.title(class_names[label])

plt.axis(False)
plt.show()'''
print(f'image size: {img.shape}')

print(f'label : {label}, label size : {label.shape}')





#model 0 :build a basline model

#a baseline ml model is a simple reference model used to evaluate whether more complex models provide real improvement.







flatten_model= nn.Flatten()

x = train_features_batch[0]



print(x.shape)

#pass through flatten

output = flatten_model(x)
print(f'shape before flattening:{x.shape} ->colorchannels,height,width')
print(f'shape after flattinging:{output.shape} -> colourchannels,height*width')
print(output.squeeze) 



from torch import nn













class FashionMNistModelV0(nn.Module):
    def __init__(self,input_shape:int,hidden_units,output_shape:int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features = input_shape,
                      out_features = hidden_units,),
            nn.Linear(in_features = hidden_units,
                      out_features = output_shape)
        )
        
    def forward(self,x):
        return self.layer_stack(x)
    
torch.manual_seed(42)
#output of flatten must = input shapes
model_0 = FashionMNistModelV0(input_shape = 784, hidden_units=10,
                            output_shape = len(class_names)
                            
    
).to('cpu')



dummy_x = torch.rand([1,1,28,28])
print(model_0(dummy_x))


#set up loss ,optim,eval metricsimport os
import requests

# Delete the old file if it exists to ensure a fresh start
if os.path.exists("helper_functions.py"):
    os.remove("helper_functions.py")

print("Downloading helper_functions.py...")
# Use the 'raw' URL to get the actual code
url = "https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py"
request = requests.get(url)

with open("helper_functions.py", "wb") as f:
    f.write(request.content)

print("Download complete!")

from helper_functions import accuracy_fn



loss_fn = torch.nn.CrossEntropyLoss()

optim = torch.optim.SGD(params = model_0.parameters(),lr = 0.01)



#machine learning is very experimental so we need to track the results
#we need to rack models performance
#models speed

from timeit import default_timer as timer

def print_train_time(start: float,
                     end: float,
                     device: torch.device = None):
    """prints difference between start and end time."""
    total_time = end - start
    print(f'Training time on {device}: {total_time:.3f} seconds')
    return total_time

starttime = timer()
end_time = timer()
print_train_time(start = starttime, end = end_time, device = 'cpu')
print_train_time(start = starttime, end = end_time, device = 'cuda')
#Training and testing loop
#train with batches of data
#loop trhough epochs
#loop through training batches, perform training steps, calculate the train lose per batch
#loop through testing batches, perform testing steps, calculate the test loss per batch
#print out whats happening
#time it for fun
from tqdm.auto import tqdm


torch.manual_seed(42)
train_time_start_on_cpu = timer()
#keep small for faster trianing time


epochs = 3
for epoch in tqdm(range(epochs)):
    print(f'Epoch: {epoch}\n-------')
    train_loss = 0
    train_acc = 0
    model_0.train()
    
    
    #we use batches for a few reasons, the reason thats my favorite is that you can train the model more with the same amount of data, because your stepping the data more times, once for each batch
    
    for batch, (X,y) in enumerate(train_dataloader):
        y_pred = model_0(X)
        loss = loss_fn(y_pred,y)
        optim.zero_grad()
        loss.backward()
        #step the optim each batch so it adjusts more
        optim.step()
        
        train_loss += loss.item()   #because we are checking batches we must sum the total lost by all of the data
        train_acc += accuracy_fn(y_true = y,
                                 y_pred = y_pred.argmax(dim=1))
        if batch % 400 == 0:
            print(f'looked at {batch} samples')
        
    train_loss /= len(train_dataloader)#calculates average loss per batch, could get total loss but not required
    train_acc /= len(train_dataloader)
    
    test_loss = 0
    test_acc = 0
       
    model_0.eval()
    
    with torch.inference_mode():
        for X,y in test_dataloader:
            test_pred = model_0(X)
            loss = loss_fn(test_pred,y)
            
            test_loss += loss.item()
            test_acc += accuracy_fn(y_true = y,
                                    y_pred = test_pred.argmax(dim=1))
            
        test_loss /= len(test_dataloader)
        test_acc /= len(test_dataloader)
        
    print(f'Train loss: {train_loss:.5f} | Train acc: {train_acc:.2f}% | Test loss: {test_loss:.5f} | Test acc: {test_acc:.2f}%\n')

    
#need to evaluate the model now


torch.manual_seed(42)
def eval_model(model :torch.nn.Module, data_loader :torch.utils.data.DataLoader,loss_fn,accuracy_fn):
    #return a dictionary containing the results of model rpediction on data_loader
    
    
    loss,acc = 0,0
    model.eval()
    with torch.inference_mode():
        for X,y in tqdm(data_loader):#must loop through with both x and y so that you can compare the predicted y with the datas correct label
            
            X,y = X.to(device),y.to(device)
            y_pred = model(X)#pass data thorugh the model
            #accumulate the loss and acc values per batch
            #argmax gets the index of the highest value along a given dimenson
            loss += loss_fn(y_pred,y)
            acc += accuracy_fn(y_true = y, y_pred = y_pred.argmax(dim=1))
            
            #scale loss and acc to find the average loss/acc per batch
            
        loss /= len(data_loader)
        acc /= len(data_loader)
    return {"model_name:":model.__class__.__name__,#only works when model was created with a class
            "model_loss":loss.item(),
            "\nmodel.acc":acc}
    
    
#calculate model 0 results on test dataset

model_0_results = eval_model(model = model_0,data_loader = test_dataloader,loss_fn = loss_fn, accuracy_fn = accuracy_fn)
            
            
print(model_0_results)








#setup device agnostic code for using a gpu if there is one

device = 'cuda' if torch.cuda.is_available() else 'cpu'




#now lets make a model with non linearity
#model_1


#ie relu






class FashionMNISTModelV1(nn.Module):
    def __init__(self, input_shape:int,hidden_shape:int,output_shape:int):
        
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features = input_shape,out_features = hidden_shape),
            nn.ReLU(),
            nn.Linear(in_features = hidden_shape,out_features = output_shape),
            nn.ReLU()
                      )
            
            
    def forward(self,x:torch.Tensor):
        return self.layer_stack(x)
    
    

        
        
torch.manual_seed(42)
model_1 = FashionMNISTModelV1(784,10,len(class_names)).to(device)

print(next(model_0.parameters()).device)
print(next(model_1.parameters()).device)


loss_fn = nn.CrossEntropyLoss()
optim = torch.optim.SGD(params = model_1.parameters(),lr = 0.01)



#training and testing loop
#gonna functionise this


        
        
def train_step(model: torch.nn.Module,data_loader:torch.utils.data.DataLoader,loss_fn:torch.nn.Module,optim : torch.optim.Optimizer,accuracy_fn,device: torch.device = device):

    train_loss,train_acc = 0,0
    for batch, (X,y) in enumerate(train_dataloader):
        #put data on target device
        X,y = X.to(device),y.to(device)
        
        y_pred = model(X)
        loss = loss_fn(y_pred,y)
        optim.zero_grad()
        loss.backward()
        #step the optim each batch so it adjusts more
        optim.step()
        
        train_loss += loss.item()   #because we are checking batches we must sum the total lost by all of the data
        train_acc += accuracy_fn(y_true = y,
                                 y_pred = y_pred.argmax(dim=1))
        if batch % 400 == 0:
            print(f'looked at {batch} samples')
    
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f'training loss:{train_loss:.3f}, training acc:{train_acc:.2f}%')
    
    
    
def test_step(model:torch.nn.Module,data_loader:torch.utils.data.DataLoader,loss_fn:torch.nn.Module,accuracy_fn,device: torch.device = device):
    test_loss,test_acc = 0,0
    
    
    model.eval()
    
    
    with torch.inference_mode():
        
        for X,y in data_loader:
            #send to target dvice
            
            X,y = X.to(device),y.to(device)
            
            test_pred = model(X)
            
            
            test_loss += loss_fn(test_pred,y)
            
            test_acc += accuracy_fn(y_true = y, y_pred = test_pred.argmax(dim = 1))
            
            #adjust metrics and print out
            
            
        test_loss /= len(data_loader)
        
        test_acc /= len(data_loader)
        
        print(f'test loss:{test_loss:.3f}, test acc:{test_acc:.2f}%')
        
        
        
torch.manual_seed(42)


from timeit import default_timer as timer



train_time_start_on_gpu = timer()


epochs =3


#create optim and eval loop


for epoch in tqdm(range(epochs)):
    
    print(f'epoch:{epoch}-------')
    
    
    train_step(model_1,data_loader=train_dataloader,loss_fn = loss_fn,optim = optim,accuracy_fn = accuracy_fn,device = device)
    
    
    test_step(model_1,data_loader=test_dataloader,loss_fn = loss_fn,accuracy_fn = accuracy_fn,device =device)
    

    
train_time_end_on_gpu = timer()

total_train_time_model_1 = print_train_time(start = train_time_start_on_gpu,end = train_time_end_on_gpu,device = device)



#on average non linearity is actually less accurate than linear


#how to make deep learning go Brrr from first principles




#get model 1 results dictionary



model_1_results = eval_model(model = model_1,data_loader = test_dataloader,loss_fn = loss_fn,accuracy_fn = accuracy_fn)


print('\n' * 10)
print(f'model_0 results{model_0_results}')
print(f'model_1 results{model_1_results}')

print('this clearly shows that linearity at least with this dataset is far more accurate than linearity')








