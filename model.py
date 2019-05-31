import torch
import torch.nn.functional as F
from torch import nn
from ml_toolkit.pytorch_utils.misc import make_variable

def get_hash_function(enc,h):

    def hash_func(images):
        images = make_variable(images.float())
        basic_feats = enc(images)
        code = torch.sign(h(basic_feats))
        return code.cpu()

    return hash_func

class LeNetEncoder(nn.Module):

    def __init__(self):
        """Init LeNet encoder."""
        super(LeNetEncoder, self).__init__()

        self.restored = False

        self.encoder = nn.Sequential(
            # 1st conv layer
            # input [3 x 28 x 28]
            # output [20 x 12 x 12]
            nn.Conv2d(3, 20, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            # 2nd conv layer
            # input [20 x 12 x 12]
            # output [50 x 4 x 4]
            nn.Conv2d(20, 50, kernel_size=5),
            nn.Dropout2d(),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU()
        )
        self.fc1 = nn.Linear(50 * 4 * 4, 500)

    def forward(self, input):
        """Forward the LeNet."""
        conv_out = self.encoder(input)
        feat = self.fc1(conv_out.view(-1, 50 * 4 * 4))
        return feat


class LeNetCodeGen(nn.Module):

    def __init__(self,code_len):
        super(LeNetCodeGen, self).__init__()
        self.fc1 = nn.Linear(500, 100)
        self.fc1_bnm = nn.BatchNorm1d(100)
        self.fc2 = nn.Linear(100, code_len)

    def forward(self, feat):
        out = F.relu(feat)
        out = F.sigmoid(self.fc1_bnm(self.fc1(out)))
        out = self.fc2(out)
        return F.tanh(out)

class Discriminator(nn.Module):

    def __init__(self, input_dims, hidden_dims, output_dims):
        """Init discriminator."""
        super(Discriminator, self).__init__()

        self.restored = False

        self.layer = nn.Sequential(
            nn.Linear(input_dims, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, output_dims),
            nn.Softmax()
        )

    def forward(self, input):
        """Forward the discriminator."""
        out = self.layer(input)
        return out
