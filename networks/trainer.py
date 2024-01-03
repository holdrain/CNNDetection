import functools
import torch
import torch.nn as nn
from networks.resnet import resnet50
from networks.base_model import BaseModel, init_weights
from accelerate import Accelerator

class Trainer(BaseModel):
    def name(self):
        return 'Trainer'

    def __init__(self, opt, accelerator,model,optimizer):
        super(Trainer, self).__init__(opt)

        self.accelerator = accelerator
        self.device = accelerator.device
        if self.isTrain:
            self.model = model
            if opt.arch == "resnet":
                torch.nn.init.normal_(self.model.fc.weight.data, 0.0, opt.init_gain)
            self.model = self.accelerator.prepare(model)

            self.loss_fn = nn.CrossEntropyLoss()
            self.optimizer = self.accelerator.prepare(optimizer)

        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.epoch)



    def adjust_learning_rate(self, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] /= 10.
            if param_group['lr'] < min_lr:
                return False
        return True

    def set_input(self, input):
        self.input = input[0]
        self.label = input[1].float()


    def forward(self):
        self.output = self.model(self.input)

    def get_loss(self):
        return self.loss_fn(self.output.squeeze(1), self.label.long())

    def optimize_parameters(self):
        self.forward()
        self.loss = self.loss_fn(self.output.squeeze(1), self.label.long())
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

