from PIL import Image
import torch
import torchvision.transforms.functional as TF

from MNISTNet import MNISTNet

#initialising a MNISTNet model instance
val_model = MNISTNet()
val_model.load_state_dict(torch.load('./results/model.pth'))

#loading and preprocessing a sample image
img = Image.open('./sample_seven.jpg').convert('L')
x = TF.to_tensor(img)
x.unsqueeze_(0)

#making a prediction and displaying the result
outputs = val_model(x)
_, predicted = torch.max(outputs, 1)
print('Predicted number is {}'.format(predicted[0]))
