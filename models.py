import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, seq_len, input_dim, hidden_dim, num_node, strategy):
        super().__init__()

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(seq_len * input_dim, num_node)
        self.fc2 = nn.Linear(num_node, num_node)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.z_loc = nn.Linear(num_node, hidden_dim)
        self.strategy = strategy
        if strategy != 'Deterministic':
            self.z_logit_scale = nn.Linear(num_node, hidden_dim)

    def forward(self, sequence):  # sequence has dimension (batch_size, seq_len, input_dim)
        if self.training:
            raise ('This is only for inferencing. Set model.eval()')

        else:
            z = self.flatten(sequence)
            z = self.dropout(self.relu(self.fc1(z)))
            z = self.fc2(z)
            z_loc = self.z_loc(z)

            return z_loc

class Classifier(nn.Module):
    def __init__(self, hidden_dim, num_node, out_node):
        super().__init__()
        self.out_node = out_node
        self.fc1 = nn.Linear(hidden_dim, num_node)
        self.fc2 = nn.Linear(num_node, out_node)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        if out_node == 1:
            self.sigm = nn.Sigmoid()
        else:
            self.lsoftm = nn.LogSoftmax(dim=1)

    def forward(self, z): # z has dimension (batch_size, hidden_dim)
        z = self.dropout(self.relu(self.fc1(z)))

        if self.out_node == 1:
            output = self.sigm(self.fc2(z))
        else:
            output = self.lsoftm(self.fc2(z))
        return output