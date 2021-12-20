import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, encoder_hidden_dim, decoder_hidden_dim):
        super().__init__()

        # hidden vector
        self.attn_hidden_vector = nn.Linear(encoder_hidden_dim , encoder_hidden_dim+decoder_hidden_dim)

        #attention scoring function
        self.attn_scoring_fn = nn.Linear(encoder_hidden_dim+decoder_hidden_dim, encoder_hidden_dim+decoder_hidden_dim, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden = [1, batch size, decoder hidden dim]
        src_len = encoder_outputs.shape[0]

        hidden = hidden.repeat(src_len, 1, 1)

        # Calculate Attention Hidden values
        attn_hidden = torch.tanh(self.attn_hidden_vector(torch.cat((hidden, encoder_outputs), dim=1)))

        # Calculate the Scoring function
        attn_scoring_vector = self.attn_scoring_fn(attn_hidden)

        # Softmax function 
        return F.softmax(attn_scoring_vector, dim=1)
