import os
import torch
from torch import nn
import torchvision 
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





class FashionMNISTV2(nn.Module):
    
    #model architecture that replicates tinyVGG
    
    def __init__(self,input_shape:int,hidden_units:int,output_shape:int):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            #2ddata
            nn.Conv2d(in_channels= input_shape,out_channels = hidden_units,kernel_size = 3, stride = 1,padding = 1),
            #values we set ourselves in our nn are called hyperparameters,
            nn.ReLU(),
            nn.Conv2d(in_channels= hidden_units,out_channels = hidden_units,kernel_size = 3, stride = 1,padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels = hidden_units,out_channels = hidden_units,kernel_size = 3,stride = 1,padding = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels = hidden_units,out_channels = hidden_units,kernel_size = 3, stride = 1,padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features = hidden_units *7*7,out_features = output_shape))
        
    def forward(self,x):

        x = self.conv_block_1(x)
        #print(f'output_shape of conv-block_1:{x.shape}')
        x = self.conv_block_2(x)
        #print(f'output_shape of conv-block_2:{x.shape}')
        x = self.classifier(x)
        #print(f'output_shape of classifier:{x.shape}')
        return x 
        
        
#instantiate

torch.manual_seed(42)


model_2 = FashionMNISTV2(input_shape = 1,hidden_units = 10,output_shape = len(class_names)).to(device)



#step through nn.Conv2d

#dummy check


torch.manual_seed(42)

#create a batch of images

images = torch.randn(size = (32,3,64,64))
test_image = images[0]
print(f'image batch shape:{images.shape}')
print(f'single imgae shape:{test_image.shape}')
print(f'test number{test_image}')

#create a single conv2d layer
                        #same as colour cahnnels of your shape(3)
                                        #equivalent to hidden units
                                                        #same as (3,3)







#steppinng through nn.maxpool2d()



conv_layer = nn.Conv2d(in_channels = 3,out_channels = 10,kernel_size= 3,stride =1,padding =0)

conv_output = conv_layer(test_image)





pool_layer  = nn.MaxPool2d(kernel_size = 2)

afterpool = pool_layer(conv_output)


print(f'test image through conv layer{conv_output}')
print(f' test image after pool layer{afterpool}')
print(f'test image original shape{test_image.shape}')
print(f'test image shape{conv_output.shape}')
print(f'test image shape{afterpool.shape}')



#create random tensor with similar number of dimensions

torch.manual_seed(42)


random_tensor = torch.rand(size = (1,1,2,2))
print(f'\n random_tensor : {random_tensor}')
print(f'random_tensor.shape:{random_tensor.shape}')

max_pool_layer = nn.MaxPool2d(kernel_size= 2)

max_pool_tensor = max_pool_layer(random_tensor)
print(f'\n Max pool tensor: \n {max_pool_tensor}')
print(f'\n Max pool tensor shape: { max_pool_tensor.shape}')#so it really compressees it


#forward pass
rand_image_tensor = torch.randn(size = (1,1,28,28))

print(rand_image_tensor.shape)


print(model_2(rand_image_tensor.to(device)))


#to checkk for errors in your model, its good to step thorugh each individuall layer



#setup loss function and optimizer, eval metrics


def accuracy_fn(y_true, y_pred):
    """Calculates accuracy percentage."""
    correct = torch.eq(y_true, y_pred).sum().item()
    return (correct / len(y_pred)) * 100


loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(params = model_2.parameters(),lr = 0.01)


print(model_2.state_dict)



#train adn test 

torch.manual_seed(42)
torch.cuda.manual_seed(42)





def train_step(model: torch.nn.Module,data_loader:torch.utils.data.DataLoader,loss_fn:torch.nn.Module,optim : torch.optim.Optimizer,accuracy_fn,device: torch.device = device):
    model.train()
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
        
        

def print_train_time(start: float,
                     end: float,
                     device: torch.device = None):
    """prints difference between start and end time."""
    total_time = end - start
    print(f'Training time on {device}: {total_time:.3f} seconds')
    return total_time
        

from tqdm.auto import tqdm

from timeit import default_timer as timer
train_time_start_model_2 = timer()
epochs = 3












for epoch in tqdm(range(epochs)):

    print(f'epoch: {epoch}\n---')
    train_step(model = model_2,data_loader = train_dataloader,loss_fn = loss_fn,optim = optimizer,accuracy_fn=accuracy_fn,device = device)

    
    test_step(model = model_2,data_loader = test_dataloader,accuracy_fn = accuracy_fn,device = device,loss_fn=loss_fn)
    
train_time_end_model_2= timer()



total_train_time_model_2 = print_train_time(start = train_time_start_model_2,end = train_time_end_model_2)






#get a resutlts dictionary
torch.manual_seed(42)
def eval_model(model :torch.nn.Module, data_loader :torch.utils.data.DataLoader,loss_fn,accuracy_fn,device):
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
    
model_2_results = eval_model(
    model = model_2,
    data_loader = test_dataloader,
    loss_fn = loss_fn,
    accuracy_fn = accuracy_fn,
    device = device
    
)


print(model_2_results)







def make_predictions(model: torch.nn.Module, data: list, device: torch.device = device):
    pred_probs = []
    model.eval()
    with torch.inference_mode():
        for sample in data:
            # Prepare sample: add a batch dimension and move to device
            # (3, 64, 64) -> (1, 3, 64, 64)
            sample = torch.unsqueeze(sample, dim=0).to(device) 

            # Forward pass (model outputs raw logits)
            pred_logit = model(sample)

            # Get prediction probability (logit -> probability)
            pred_prob = torch.softmax(pred_logit.squeeze(), dim=0)

            # Perform prediction on CPU for Matplotlib later
            pred_probs.append(pred_prob.cpu())
            
    # Stack the list of tensors into a single tensor
    return torch.stack(pred_probs)


import random
random.seed(42)
test_samples = []
test_labels = []

for sample,label in random.sample(list(test_data),k=9):


    test_samples.append(sample)
    test_labels.append(label)



print(test_samples[0].shape)







plt.imshow(test_samples[0].squeeze(),cmap = 'gray')
plt.title(class_names[test_labels[0]])
plt.axis(False)
plt.show()






pred_probs = make_predictions(model = model_2,data = test_samples,device = device)
print(pred_probs[:2])
#conmvert pred probs to lavels
pred_classes = pred_probs.argmax(dim = 1)
print(pred_classes)




#plot the images and the predictions
plt.figure(figsize = (9,9))
nrows = 3
ncols = 3

for i,sample in enumerate(test_samples):
    plt.subplot(nrows, ncols, i+1)
    
    #find the prediction label in text form, eg sandal
    pred_label = class_names[pred_classes[i]]
    
    #get the true label
    truth_label = class_names[test_labels[i]]
    
    #create a title for the plot
    title_text = f'pred: {pred_label} | truth: {truth_label}'
    
    #check for equality between pred and truth and change colour of title text
    if pred_label == truth_label:
        plt.title(title_text, fontsize=10, c='g')  # green if correct
    else:
        plt.title(title_text, fontsize=10, c='r')  # red if wrong
        
    plt.imshow(sample.squeeze(), cmap='gray')  # also: 'grey' -> 'gray' (standard spelling)
    plt.axis(False)  # turn off axis for cleaner look

plt.show() 
    
    
    
    
    
#confusion matrix




#to do this we must
#make predictions, with our trained model onthe test dataset
#look at torchmetrics
#plot the confusion matrix using 'mlextend.plotting.plot_confusion_matrix'


import mlxtend


#import tqdm.auto for progress bar, already done this




#1.make predictions
y_preds = []

model_2.eval()
with torch.inference_mode():
    for X,y in tqdm(test_dataloader,desc = 'making predictions . . .'):
        #send the data and targets to target device
        X,y = X.to(device), y.to(device)
        
        y_logit = model_2(X)
    
    #turn predictions to logits - > pred probs - > pred labels(argmax
    
    
        y_pred = torch.softmax(y_logit.squeeze(),dim = 0).argmax(dim = 1)
    
    #put preds in cpu for eval
    #matplot is on cpu
    
        y_preds.append(y_pred.cpu())
print(y_preds)

y_pred_tensor = torch.cat(y_preds)#concatenate list of predctions to a tensor

print(y_pred_tensor[:10])



print(len(y_pred_tensor))


import torchmetrics
from torchmetrics import ConfusionMatrix


#see metrics












confmat = ConfusionMatrix(num_classes = len(class_names),task = 'multiclass')




conf_mat_tensor = confmat(preds = y_pred_tensor,target = test_data.targets)



from mlxtend.plotting import plot_confusion_matrix
#2



fig,ax = plot_confusion_matrix(
    conf_mat = conf_mat_tensor.numpy(), #matplotlib likes numpy
    class_names = class_names,
    figsize = (10,7)
)





#save and load model meow

from pathlib import Path
#create model directory path

MODEL_PATH =Path('models')
MODEL_PATH.mkdir(parents = True,exist_ok = True)

MODEL_NAME = '03_Pytorch_computervision_model_2.pth'


MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

#save modell state dict

print(f'saving model to:{MODEL_SAVE_PATH}')
torch.save(obj =model_2.state_dict(),f = MODEL_SAVE_PATH)


import os
print(f"Your current folder is: {os.getcwd()}")
print(f"The file should be at: {MODEL_SAVE_PATH.absolute()}")
print(f"Does the file exist? {MODEL_SAVE_PATH.exists()}")



