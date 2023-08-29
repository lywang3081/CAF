import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

__all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}

class AlexNet(nn.Module):

    def __init__(self, taskcla):
        super(AlexNet, self).__init__()
        self.taskcla = taskcla
        self.nLearner = 5
        self.f_size = 34
        self.s_gate = 1

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.dropout = nn.Dropout()
        self.sig_gate = torch.nn.Sigmoid()

        self.num_tasks = 0
        self.last = torch.nn.ModuleList()
        for t, n in self.taskcla:
            self.num_tasks += 1
            self.last.append(torch.nn.Linear(self.f_size * 64, n))

        self.net1 = nn.Sequential(
            nn.Conv2d(3, self.f_size, kernel_size=11, stride=4, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(self.f_size, self.f_size*3, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(self.f_size*3, self.f_size*6, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.f_size*6, self.f_size*4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.f_size*4, self.f_size*4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.AdaptiveAvgPool2d((6, 6))
        )

        self.fc1 = nn.Linear(self.f_size * 4 * 6 * 6, self.f_size * 64)

        self.net2 = nn.Sequential(
            nn.Conv2d(3, self.f_size, kernel_size=11, stride=4, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(self.f_size, self.f_size*3, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(self.f_size*3, self.f_size*6, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.f_size*6, self.f_size*4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.f_size*4, self.f_size*4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.AdaptiveAvgPool2d((6, 6))
        )

        self.fc2 = nn.Linear(self.f_size * 4 * 6 * 6, self.f_size * 64)

        self.net3 = nn.Sequential(
            nn.Conv2d(3, self.f_size, kernel_size=11, stride=4, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(self.f_size, self.f_size*3, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(self.f_size*3, self.f_size*6, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.f_size*6, self.f_size*4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.f_size*4, self.f_size*4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.AdaptiveAvgPool2d((6, 6))
        )

        self.fc3 = nn.Linear(self.f_size * 4 * 6 * 6, self.f_size * 64)

        self.net4 = nn.Sequential(
            nn.Conv2d(3, self.f_size, kernel_size=11, stride=4, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(self.f_size, self.f_size*3, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(self.f_size*3, self.f_size*6, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.f_size*6, self.f_size*4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.f_size*4, self.f_size*4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.AdaptiveAvgPool2d((6, 6))
        )

        self.fc4 = nn.Linear(self.f_size * 4 * 6 * 6, self.f_size * 64)

        self.net5 = nn.Sequential(
            nn.Conv2d(3, self.f_size, kernel_size=11, stride=4, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(self.f_size, self.f_size*3, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(self.f_size*3, self.f_size*6, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.f_size*6, self.f_size*4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.f_size*4, self.f_size*4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.AdaptiveAvgPool2d((6, 6))
        )

        self.fc5 = nn.Linear(self.f_size * 4 * 6 * 6, self.f_size * 64)

    def forward(self, x, t=0, avg_act=False, return_expert=False):

        self.Experts = []

        h1 = self.net1(x)
        h1 = h1.view(x.shape[0], -1)
        h1 = self.relu(self.fc1(self.dropout(h1)))
        self.Experts.append(h1.unsqueeze(0))

        h2 = self.net2(x)
        h2 = h2.view(x.shape[0], -1)
        h2 = self.relu(self.fc2(self.dropout(h2)))
        self.Experts.append(h2.unsqueeze(0))

        h3 = self.net3(x)
        h3 = h3.view(x.shape[0], -1)
        h3 = self.relu(self.fc3(self.dropout(h3)))
        self.Experts.append(h3.unsqueeze(0))

        h4 = self.net4(x)
        h4 = h4.view(x.shape[0], -1)
        h4 = self.relu(self.fc4(self.dropout(h4)))
        self.Experts.append(h4.unsqueeze(0))

        h5 = self.net5(x)
        h5 = h5.view(x.shape[0], -1)
        h5 = self.relu(self.fc5(self.dropout(h5)))
        self.Experts.append(h5.unsqueeze(0))

        h = torch.cat([h_result for h_result in self.Experts], 0)
        h = torch.sum(h, dim=0).squeeze(0)

        y = []
        for t,i in self.taskcla:
            y.append(self.last[t](h))

        self.grads = {}
        self.act = []

        def save_grad(name):
            def hook(grad):
                self.grads[name] = grad

            return hook

        if avg_act == True:
            names = [0, 1, 2, 3, 4, 5, 6]
            act = [act1, act2, act3, act4, act5, act6, act7]

            self.act = []
            for i in act:
                self.act.append(i.detach())
            for idx, name in enumerate(names):
                act[idx].register_hook(save_grad(name))

        if return_expert:
            self.Experts_y = []
            for i in range(self.nLearner):
                h_exp = self.Experts[i].squeeze(0)

                # using joint classifier
                y_exp = self.last[t](h_exp)
                self.Experts_y.append(y_exp)

            return y, self.Experts_y, self.Experts

        else:
            return y


def alexnet(taskcla, pretrained=False):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexNet(taskcla)
    
    if pretrained:
        pre_model = torchvision.models.alexnet(pretrained=True)
        for key1, key2 in zip(model.state_dict().keys(), pre_model.state_dict().keys()):
            if 'last' in key1:
                break
            if model.state_dict()[key1].shape == torch.tensor(1).shape:
                model.state_dict()[key1] = pre_model.state_dict()[key2]
            else:
                model.state_dict()[key1][:] = pre_model.state_dict()[key2][:]
    
    return model
