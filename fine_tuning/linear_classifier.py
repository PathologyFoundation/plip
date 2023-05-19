import torch.nn as nn

# Define a linear classifier
class LinearClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)
        
    def forward(self, x):
        # Convert input matrix to the same data type as self.weight
        x = x.to(self.fc.weight.dtype)
        out = self.fc(x)
        return out