import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


from torchvision.models import efficientnet

class EfficientNetForecast(nn.Module):
    def __init__(self, num_classes=8, compound_coef=0):
        super(EfficientNetForecast, self).__init__()
        self.backbone = efficientnet.EfficientNet.from_pretrained(f'efficientnet-b{compound_coef}', num_classes=num_classes)

    def forward(self, x):
        return self.backbone(x)


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

        # Assuming input size of [batch, 3, 8, 16]
        height = 8  # Adjust based on your input dimensions
        width = 16  # Adjust based on your input dimensions

        # Create a dummy input to calculate the size after conv layers
        dummy_input = torch.autograd.Variable(torch.zeros(1, 3, height, width))
        dummy_output = self.forward_conv_layers(dummy_input)
        flattened_size = dummy_output.data.reshape(1, -1).size(1)

        self.fc1 = nn.Linear(flattened_size, 100)
        self.fc2 = nn.Linear(100, 8)  # Adjusted to match the expected output size

    def forward_conv_layers(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        return x

    def forward(self, x):
        x = self.forward_conv_layers(x)
        # Use .reshape() to ensure compatibility even with non-contiguous tensors
        x = x.reshape(-1, x.size(1) * x.size(2) * x.size(3))
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class batchNormCNN(nn.Module):
    def __init__(self):
        super(batchNormCNN, self).__init__()
        # Define the first convolutional layer and batch normalization
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        # Define the second convolutional layer and batch normalization
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        # Define the third convolutional layer and batch normalization
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

        # Use a dummy input to calculate the flattened size dynamically
        dummy_input = torch.zeros(1, 3, 8, 16)  # Adjust the dimensions (1, C, H, W) as per your input
        dummy_output = self.forward_conv_layers(dummy_input)
        flattened_size = dummy_output.nelement() // dummy_output.shape[0]

        # Fully connected layers
        self.fc1 = nn.Linear(flattened_size, 100)
        self.fc2 = nn.Linear(100, 8)  # Adjusted to match the expected output size

    def forward_conv_layers(self, x):
        # Apply convolutional layers with ReLU and batch normalization
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        return x

    def forward(self, x):
        x = self.forward_conv_layers(x)
        # Flatten the tensor for the fully connected layer
        x = x.reshape(-1, x.size(1) * x.size(2) * x.size(3))
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class batchNormDeepCNN(nn.Module):
    def __init__(self):
        super(batchNormDeepCNN, self).__init__()
        # Convolutional and BatchNorm layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        # Reduce the frequency of pooling to preserve spatial dimensions
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

        # Adjust dummy input size to reflect actual data size
        # Calculate the flattened size dynamically to handle changes in architecture
        dummy_input = torch.zeros(1, 3, 8, 16)
        dummy_output = self.forward_dummy(dummy_input)
        flattened_size = dummy_output.nelement() // dummy_output.shape[0]

        # Fully connected layers
        self.fc1 = nn.Linear(flattened_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 8)  # Example for 10 classes

    def forward_dummy(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)  # Apply pooling less frequently
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn5(self.conv5(x)))
        # No additional pooling to prevent reduction to too small a size
        return x

    def forward(self, x):
        x = self.forward_dummy(x)
        x = x.reshape(-1, x.size(1) * x.size(2) * x.size(3))
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class batchNormDeepDropoutCNN(nn.Module):
    def __init__(self):
        super(batchNormDeepDropoutCNN, self).__init__()
        # Convolutional and BatchNorm layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)

        # Dropout layers for convolutional layers
        self.dropout2d = nn.Dropout2d(p=0.3)  # Example dropout rate for convolutional layers

        # Reduce the frequency of pooling to preserve spatial dimensions
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

        # Adjust dummy input size to reflect actual data size
        dummy_input = torch.zeros(1, 3, 8, 15)
        dummy_output = self.forward_dummy(dummy_input)
        flattened_size = dummy_output.nelement() // dummy_output.shape[0]

        # Fully connected layers
        self.fc1 = nn.Linear(flattened_size, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 8)  # Adjusted for your output size

        # Dropout layers for fully connected layers
        self.dropout = nn.Dropout(p=0.4)  # Example dropout rate for fully connected layers

    def forward_dummy(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(self.dropout2d(x))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(self.dropout2d(x))
        x = self.relu(self.bn3(self.conv3(x)))
        #x = self.pool(self.dropout2d(x))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.pool(self.dropout2d(x))
        x = self.relu(self.bn5(self.conv5(x)))
        return x

    def forward(self, x):
        x = self.forward_dummy(x)
        x = x.reshape(-1, x.size(1) * x.size(2) * x.size(3))
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

class batchNormLeakyLessDropoutCNN(nn.Module):
    def __init__(self):
        super(batchNormLeakyLessDropoutCNN, self).__init__()
        # Convolutional and BatchNorm layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        # Dropout layers for convolutional layers
        self.dropout2d = nn.Dropout2d(p=0.3)  # Example dropout rate for convolutional layers

        # Reduce the frequency of pooling to preserve spatial dimensions
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.leaky = nn.LeakyReLU()
        # Adjust dummy input size to reflect actual data size
        dummy_input = torch.zeros(1, 3, 8, 15)
        dummy_output = self.forward_dummy(dummy_input)
        flattened_size = dummy_output.nelement() // dummy_output.shape[0]

        # Fully connected layers
        self.fc1 = nn.Linear(flattened_size, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 8)  # Adjusted for your output size

        # Dropout layers for fully connected layers
        self.dropout = nn.Dropout(p=0.3)  # Example dropout rate for fully connected layers

    def forward_dummy(self, x):
        x = self.leaky(self.bn1(self.conv1(x)))
        x = self.pool(self.dropout2d(x))
        x = self.leaky(self.bn2(self.conv2(x)))
        x = self.pool(self.dropout2d(x))
        x = self.leaky(self.bn3(self.conv3(x)))
        return x

    def forward(self, x):
        x = self.forward_dummy(x)
        x = x.reshape(-1, x.size(1) * x.size(2) * x.size(3))
        x = self.dropout(self.leaky(self.fc1(x)))
        x = self.dropout(self.leaky(self.fc2(x)))
        x = self.fc3(x)
        return x

class batchNormLeakyDropoutCNN(nn.Module):
    def __init__(self):
        super(batchNormLeakyDropoutCNN, self).__init__()
        # Convolutional and BatchNorm layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        # Dropout layers for convolutional layers
        self.dropout2d = nn.Dropout2d(p=0.3)  # Example dropout rate for convolutional layers

        # Reduce the frequency of pooling to preserve spatial dimensions
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.leaky = nn.LeakyReLU()
        # Adjust dummy input size to reflect actual data size
        dummy_input = torch.zeros(1, 3, 8, 15)
        dummy_output = self.forward_dummy(dummy_input)
        flattened_size = dummy_output.nelement() // dummy_output.shape[0]

        # Fully connected layers
        self.fc1 = nn.Linear(flattened_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 8)  # Adjusted for your output size

        # Dropout layers for fully connected layers
        self.dropout = nn.Dropout(p=0.4)  # Example dropout rate for fully connected layers

    def forward_dummy(self, x):
        x = self.leaky(self.bn1(self.conv1(x)))
        x = self.pool(self.dropout2d(x))
        x = self.leaky(self.bn2(self.conv2(x)))
        x = self.pool(self.dropout2d(x))
        x = self.leaky(self.bn3(self.conv3(x)))
        #x = self.pool(self.dropout2d(x))
        x = self.leaky(self.bn4(self.conv4(x)))
        return x

    def forward(self, x):
        x = self.forward_dummy(x)
        x = x.reshape(-1, x.size(1) * x.size(2) * x.size(3))
        x = self.dropout(self.leaky(self.fc1(x)))
        x = self.dropout(self.leaky(self.fc2(x)))
        x = self.fc3(x)
        return x

class batchNormDeepEluDropoutCNN(nn.Module):
    def __init__(self):
        super(batchNormDeepEluDropoutCNN, self).__init__()
        # Convolutional and BatchNorm layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        #self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        #self.bn5 = nn.BatchNorm2d(256)

        # Dropout layers for convolutional layers
        self.dropout2d = nn.Dropout2d(p=0.3)  # Example dropout rate for convolutional layers

        # Reduce the frequency of pooling to preserve spatial dimensions
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.leaky = nn.LeakyReLU()
        self.relu = nn.ReLU()
        self.elu  = nn.ELU()

        # Adjust dummy input size to reflect actual data size
        dummy_input = torch.zeros(1, 3, 8, 15)
        dummy_output = self.forward_dummy(dummy_input)
        flattened_size = dummy_output.nelement() // dummy_output.shape[0]

        # Fully connected layers
        self.fc1 = nn.Linear(flattened_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 8)  # Adjusted for your output size

        # Dropout layers for fully connected layers
        self.dropout = nn.Dropout(p=0.3)  # Example dropout rate for fully connected layers

    def forward_dummy(self, x):
        x = self.elu(self.bn1(self.conv1(x)))
        x = self.pool(self.dropout2d(x))
        x = self.elu(self.bn2(self.conv2(x)))
        x = self.pool(self.dropout2d(x))
        x = self.elu(self.bn3(self.conv3(x)))
        #x = self.pool(self.dropout2d(x))
        x = self.elu(self.bn4(self.conv4(x)))
        #x = self.pool(self.dropout2d(x))
        #x = self.elu(self.bn5(self.conv5(x)))
        return x

    def forward(self, x):
        x = self.forward_dummy(x)
        x = x.reshape(-1, x.size(1) * x.size(2) * x.size(3))
        x = self.dropout(self.leaky(self.fc1(x)))
        x = self.dropout(self.leaky(self.fc2(x)))
        x = self.dropout(self.leaky(self.fc3(x)))
        x = self.fc4(x)
        return x

class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers, drop_rate):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(self._make_layer(in_channels + i * growth_rate, growth_rate, drop_rate))

    def _make_layer(self, in_channels, growth_rate, drop_rate):
        layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, growth_rate, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Dropout2d(drop_rate)
        )
        return layer

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_features = layer(torch.cat(features, 1))
            features.append(new_features)
        return torch.cat(features, 1)

class TransitionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, drop_rate):
        super(TransitionBlock, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.bn(x))
        x = self.conv(x)
        x = self.pool(x)
        return x

class DenseNet(nn.Module):
    def __init__(self, num_classes=8):
        super(DenseNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def densenet(num_classes=8):
    return DenseNet(num_classes=num_classes)

class ImprovedCNN(nn.Module):
    def __init__(self):
        super(ImprovedCNN, self).__init__()
        # Convolutional and BatchNorm layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        # Dropout layers for convolutional layers
        self.dropout2d = nn.Dropout2d(p=0.3)

        # Reduce the frequency of pooling to preserve spatial dimensions
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.leaky = nn.LeakyReLU()

        # Fully connected layers
        self.fc1 = nn.Linear(256 * 1 * 2, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 8)

        # Dropout layers for fully connected layers
        self.dropout = nn.Dropout(p=0.4)

    def forward(self, x):
        x = self.leaky(self.bn1(self.conv1(x)))
        x = self.pool(self.dropout2d(x))
        x = self.leaky(self.bn2(self.conv2(x)))
        x = self.pool(self.dropout2d(x))
        x = self.leaky(self.bn3(self.conv3(x)))
        x = self.pool(self.dropout2d(x))
        x = self.leaky(self.bn4(self.conv4(x)))
        x = x.view(-1, 256 * 1 * 2)
        x = self.dropout(self.leaky(self.fc1(x)))
        x = self.dropout(self.leaky(self.fc2(x)))
        x = self.fc3(x)
        return x

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, dropout_prob=0.5):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.dropout1 = nn.Dropout2d(p=dropout_prob)  # Dropout layer after the first convolutional layer
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout2 = nn.Dropout2d(p=dropout_prob)  # Dropout layer after the second convolutional layer

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.dropout1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout2(out)

        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=8, dropout_prob=0.5):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, dropout_prob=dropout_prob)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dropout_prob=dropout_prob)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dropout_prob=dropout_prob)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dropout_prob=dropout_prob)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1, dropout_prob=0.5):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, dropout_prob))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, dropout_prob=dropout_prob))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def resnet18_with_dropout(dropout_prob=0.5):
    return ResNet(BasicBlock, [2, 2, 2, 2], dropout_prob=dropout_prob)

def resnet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])



def get_model(name):
    if name == "simple_cnn":
        return SimpleCNN()
    elif name == "batchNormLeakyDropoutCNN":
        return batchNormLeakyDropoutCNN()
    elif name == "batchNormLeakyLessDropoutCNN":
        return batchNormLeakyLessDropoutCNN()
    # Add more architectures as needed
    elif name == "batchNormCNN":
        return batchNormCNN()
    elif name == "batchNormDeepEluDropoutCNN":
        return batchNormDeepEluDropoutCNN()
    elif name == "batchNormDeepCNN":
        return batchNormDeepCNN()
    elif name == "batchNormDeepDropoutCNN":
        return batchNormDeepDropoutCNN()
    elif name == "densenet":
        return DenseNet()
    elif name == "ImprovedCNN":
        return ImprovedCNN()
    elif name == "resnet18":
        return resnet18()
    elif name =="EfficientNetForecast":
        return EfficientNetForecast()
    elif name == "resnet18_with_dropout":
        return resnet18_with_dropout()
    else:
        raise ValueError("Model not recognized.")

