import torch.nn as nn

class Pothole_RCNN(nn.Module):
    def __init__(self, num_classes, backbone):
        super(Pothole_RCNN, self).__init__()

        self.pretrained = nn.Sequential(*list(backbone.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # self.classification = nn.Sequential(nn.Flatten(), nn.Linear(512, num_classes))
        self.classification = nn.Sequential(nn.Flatten(), nn.Linear(2048, num_classes))

    def forward(self, x):

        x = self.pretrained(x)
        x = self.avgpool(x)

        class_probs = self.classification(x)

        return class_probs
