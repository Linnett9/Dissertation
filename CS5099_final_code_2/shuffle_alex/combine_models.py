# models.py

import torch
import torch.nn as nn
import torchvision.models as models

#print("Loading models")

import torch
import torch.nn as nn
import torchvision.models as models

class CombinedModel(nn.Module):
    def __init__(self, hp):
        super(CombinedModel, self).__init__()
        self.hp = hp

        self.shufflenet = models.shufflenet_v2_x1_0(pretrained=True)
        self.shufflenet.fc = nn.Identity()
        self.shufflenet_features = self.shufflenet.conv5[0].out_channels
        
        self.alexnet = models.alexnet(pretrained=True)
        self.alexnet.classifier = nn.Sequential(*list(self.alexnet.classifier.children())[:-1])
        alexnet_features = self.alexnet.classifier[-1].in_features if isinstance(self.alexnet.classifier[-1], nn.Linear) else self.alexnet.classifier[-2].in_features

        self.first_classifier = nn.Linear(self.shufflenet_features, 1)
        self.cond_activation_threshold = hp.cond_activation_threshold
        self.cond_activation_weight = hp.cond_activation_weight

        combined_features = self.shufflenet_features + alexnet_features
        self.fc1 = nn.Linear(combined_features, hp.filters)
        self.dropout = nn.Dropout(hp.dropout_rate)
        self.fc2 = nn.Linear(hp.filters, hp.num_dense_units)
        self.final_classifier = nn.Linear(hp.num_dense_units, 2)  # Two output neurons

        # Freeze all layers initially
        for param in self.shufflenet.parameters():
            param.requires_grad = False

        for param in self.alexnet.parameters():
            param.requires_grad = False

        # Only allow the final classifier to be trainable initially
        for param in self.final_classifier.parameters():
            param.requires_grad = True

    def forward(self, x):
        shufflenet_features = self.shufflenet(x)
        alexnet_features = self.alexnet(x)
        
        first_classifier_output = torch.sigmoid(self.first_classifier(shufflenet_features))
        
        modified_alexnet_features = alexnet_features.clone()
        for i in range(first_classifier_output.size(0)):
            if first_classifier_output[i].item() < self.cond_activation_threshold:
                modified_alexnet_features[i] = alexnet_features[i] * self.cond_activation_weight
        
        combined_features = torch.cat((shufflenet_features, modified_alexnet_features), dim=1)
        
        x = self.fc1(combined_features)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.dropout(x)
        final_output = self.final_classifier(x)

        return final_output


    
#print("Models loaded")