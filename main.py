import torch
import torch.nn as nn
import torch.nn.functional as F
class Model(nn.Module):
     def __init__(self,in_features=4,h1=8,h2=9,out_features=3):
         super().__init__()
         self.fc1=nn.Linear(in_features,h1)
         self.fc2=nn.Linear(h1,h2,)
         self.out=nn.Linear(h2,out_features)
     def forward(self,x):
         x=F.relu(self.fc1(x))
         x=F.relu(self.fc2(x))
         x=self.out(x)
         return x
model=Model()
model.load_state_dict(torch.load('My_iris_Model.pt'))
def prediction():
    while True:
        print('<<< Give Your Description >>>')
        sepal_len=float(input('Sepal length :'))
        sepal_wid = float(input('Sepal width :'))
        petal_len=float(input('Petal length :'))
        petal_wid = float(input('Petal width :'))
        data=torch.tensor([sepal_len,sepal_wid,petal_len,petal_wid])
        with torch.no_grad():
            prediction=model(data)
        if prediction.argmax().item()==0:
            print('Your given description says it is Setosa')
        elif prediction.argmax().item()==1:
            print('Your given description says it is Versicolor')
        else:
            print('Your given description says it is Virginica')
try:
    prediction()
except:
    prediction()