import torch
import torch.nn as nn
import torch.nn.functional as F
class NN(nn.Module):
    def __init__(self,in_features=4,h1=10,h2=20,h3=15,h4=7,out_features=3):
        super().__init__()
        self.fc1=nn.Linear(in_features,h1)
        #h3=15
        #h4=7
        self.fc2=nn.Linear(h1,h2)
        self.fc3=nn.Linear(h2,h3)
        self.fc4=nn.Linear(h3,h4)
        self.fc5=nn.Linear(h2,out_features)
    # def array_to_tensor(self,x):
    #     return torch.tensor(x)

    # def predict(self, x):
    #     x=self.x
    #     with torch.no_grad():
    #         NN.forward(x)
    def predict(self,x):
        x=torch.tensor(x)
        with torch.no_grad():
            x=F.relu(self.fc1(x))
            x=F.relu(self.fc2(x))
            x=F.relu(self.fc3(x))
            x=F.relu(self.fc4(x))
            x=self.fc5(x)
            if x.argmax().item()==0:
                prediction='Provided data says this flower is : Setosa.'
            elif x.argmax().item()==1:
                prediction='Provided data says this flower is : Versicolor.'
            else:
                prediction='Provided data says this flower is : Virginica.'
        return prediction
model=NN()
model.load_state_dict(torch.load('My-iris-model-more-efficient.pt'))