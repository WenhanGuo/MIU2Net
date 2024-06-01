from torch import nn

class Ensemble(nn.Module):
    def __init__(self, model1, model2, freeze1=False):
        super(Ensemble, self).__init__()
        self.model1 = model1
        self.model2 = model2
        if freeze1:
            for param in model1.parameters():
                param.requires_grad = False
        
    def forward(self, x):
        x1 = self.model1(x)
        x2 = self.model2(x1)
        return x2