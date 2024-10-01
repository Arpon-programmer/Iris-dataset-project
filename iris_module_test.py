# import torch
from iris_module import NN
# model=NN()
# model.load_state_dict(torch.load('My-iris-model-more-efficient.pt'))
data=[5.9,3.0,5.1,1.8]
# tensor_data=model.array_to_tensor(data)
model=NN()
pred=model.predict(data)
print((pred))
