

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

    def forward(self, tokens):
        """BiLSTM over document tokens.
        """
        pass
