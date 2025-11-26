#https://scholar.google.com/scholar?hl=en&as_sdt=0%2C14&q=Backpropagation+Applied+to+Handwritten+Zip+Code+Recognition&btnG=
import torch 

def generate_model():
    model = nn.Sequential([
        torch.nn.Conv2d(in_channels=3, out_channels = 32, kernel_size=3)
        torch.nn.ReLU(),
        #reduces the spacial dimension. The pooling layers looks at each 2x2 blocks of the feature map. It will take the max in that 2x2
        torch.nn.MaxPool2d(kernel_size=2,stride=2),

        torch.nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=2,stride=2),

        torch.nn.Flatten(),
        torch.nn.Linear(64*6*6,1024),
        torch.nn.ReLU(),
        torch.nn.Linear(1024,10),
    ])
    return model