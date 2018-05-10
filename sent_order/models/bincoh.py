

from torch import nn

from ..embeds import WordEmbedding


class Classifier(nn.Module):

    def __init__(self, vocab, lstm_dim, lstm_num_layers=1):

        super().__init__()

        self.embeddings = WordEmbedding(vocab)

        self.lstm = nn.LSTM(
            self.embeddings.weight.shape[1],
            lstm_dim,
            bidirectional=True,
            num_layers=lstm_num_layers,
            batch_first=True,
        )

        self.dropout = nn.Dropout()

        self.predict = nn.Sequential(
            nn.Linear(lstm_dim*4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
            nn.LogSoftmax(1),
        )

    def forward(self, docs):
        """Encode document tokens, predict.
        """
        x, sizes = pad_right_and_stack([
            self.embeddings.tokens_to_idx(tokens)
            for tokens in docs
        ])

        x = self.embeddings(x)
        x = self.dropout(x)

        x, reorder = pack(x, sizes)

        _, (hn, _) = self.lstm(x)
        hn = self.dropout(hn)

        # Cat forward + backward hidden layers.
        x = torch.cat([hn[0,:,:], hn[1,:,:]], dim=1)
        x = x[reorder]

        return self.predict(x)
