import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable


"""
There are many, many variations that can be done.

Some obvious ones are:
1) How to feed in z (for the generator type)
    - Right now i sample from gaussian and of lenght = output
    - Similar to C-RNN-GAN paper

2) How to feed in c?
    - I currently concat directly with the input
    Can pass in another FC layer and then concat

3) Q and Discriminator structure
    - They are comlpetely separate now
    but you can have a single block (RNN) and two different output heads.
    (dont forget to modify the optimizer params if you do join)
"""


class CreatorGenerator(nn.Module):
    """
    Creator type generator, takes random variable and c (concated together for now)
    Input: z noise and the last 2

    """
    def __init__(
        self, n_channels, hidden_size, encoder_dropout, device, bidirectional =True
    ):
        super(CreatorGenerator, self).__init__()

        self.input_size = n_channels # z_dim + 2 (accounting for c)
        self.hidden_size = hidden_size
        self.layers = 1
        self.dropout = encoder_dropout
        self.bi = bidirectional
        self.device = device

        #for init
        self.first_dim = self.layers*2 if self.bi else self.layers

        self.rnn = nn.LSTM(
            self.input_size,
            self.hidden_size,
            self.layers,
            dropout=self.dropout,
            bidirectional=self.bi,
            batch_first=True,
        )

        #random hyperparameter, change or tune its fine
        out_features = 50
        self.fc_output = nn.Sequential(
            nn.Linear(self.hidden_size*2 if self.bi else self.hidden_size, out_features),
            nn.Linear(out_features, 1),
        )

    def forward(self, inputs, input_lengths):

        x = pack_padded_sequence(inputs, input_lengths, enforce_sorted=False, batch_first=True)

        init_hidden, init_cell = self.init_hidden_lstm(self.first_dim, inputs.size()[0] , self.hidden_size, self.device)

        rnn_output, (_, _) = self.rnn(x, (init_hidden, init_cell))
        rnn_output = pad_packed_sequence(rnn_output, batch_first=True, padding_value=0.0)
        #TODO figure out how to feed packed directly to fc
        generated_output = self.fc_output(rnn_output[0])

        return generated_output

    @classmethod
    def init_hidden_lstm(self, first_dim, batch_size, hidden_size, device):

        h0 = torch.zeros(first_dim, batch_size, hidden_size).to(device)
        c0 = torch.zeros(first_dim, batch_size, hidden_size).to(device)

        return h0, c0


class Q_C_Detector(nn.Module):
    """
    Same for creator and converter type

    Q neural network:
    Takes in the generated output from the generator
    and detects the C variable used to condition

    An RNN with a final FC layer
    """

    def __init__( self, hidden_size, dropout, device, bidirectional =True):
        super(Q_C_Detector, self).__init__()

        self.input_size = 1 # output dim (1 here for sin output)
        self.hidden_size = hidden_size
        self.layers = 1
        self.dropout = dropout
        self.bi = bidirectional
        self.device = device

        #for init
        self.first_dim = self.layers*2 if self.bi else self.layers

        self.rnn = nn.LSTM(
            self.input_size,
            self.hidden_size,
            self.layers,
            dropout=self.dropout,
            bidirectional=self.bi,
            batch_first=True,
        )

        #random hyperparameter, change or tune its fine
        out_features = 50
        self.fc_output = nn.Sequential(
            nn.Linear(self.hidden_size*2 if self.bi else self.hidden_size, out_features),
            nn.Linear(out_features, 2),
        )

    def forward(self, generated_inputs):
        """
        The input will already be padded
        """
        batch_size = generated_inputs.size()[0]
        init_hidden, init_cell = self.init_hidden_lstm(self.first_dim, batch_size , self.hidden_size, self.device)

        rnn_output, (hidden_state, cell_state) = self.rnn(generated_inputs, (init_hidden, init_cell))
        print(hidden_state.shape)
        print(hidden_state.view(generated_inputs.size()[0], 2*self.hidden_size,1).shape)
        predicted_c = self.fc_output(hidden_state.view(batch_size, 2*self.hidden_size))

        return predicted_c

    @classmethod
    def init_hidden_lstm(self, first_dim, batch_size, hidden_size, device):

        h0 = torch.zeros(first_dim, batch_size, hidden_size).to(device)
        c0 = torch.zeros(first_dim, batch_size, hidden_size).to(device)

        return h0, c0



class Discriminator(nn.Module):
    """
    Same for both Creator and Converter types
    An RNN with a final FC layer
    """

    def __init__( self, hidden_size, dropout, device, bidirectional =True):
        super(Discriminator, self).__init__()

        self.input_size = 1 # output dim (1 here for sin output)
        self.hidden_size = hidden_size
        self.layers = 1
        self.dropout = dropout
        self.bi = bidirectional
        self.device = device

        #for init
        self.first_dim = self.layers*2 if self.bi else self.layers

        self.rnn = nn.LSTM(
            self.input_size,
            self.hidden_size,
            self.layers,
            dropout=self.dropout,
            bidirectional=self.bi,
            batch_first=True,
        )

        #random hyperparameter, change or tune its fine
        out_features = 50
        self.fc_output = nn.Sequential(
            nn.Linear(self.hidden_size*2 if self.bi else self.hidden_size, out_features),
            nn.Linear(out_features, 1),
        )

    def forward(self, generated_inputs):
        """
        The input will already be padded
        """
        batch_size = generated_inputs.size()[0]
        init_hidden, init_cell = self.init_hidden_lstm(self.first_dim, batch_size , self.hidden_size, self.device)

        rnn_output, (hidden_state, cell_state) = self.rnn(generated_inputs, (init_hidden, init_cell))

        predicted_c = self.fc_output(hidden_state.view(batch_size, 2*self.hidden_size))

        return predicted_c

    @classmethod
    def init_hidden_lstm(self, first_dim, batch_size, hidden_size, device):

        h0 = torch.zeros(first_dim, batch_size, hidden_size).to(device)
        c0 = torch.zeros(first_dim, batch_size, hidden_size).to(device)

        return h0, c0
