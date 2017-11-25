

import ujson

from itertools import islice
from tqdm import tqdm
from glob import glob
from collections import Counter

from gensim.models import KeyedVectors

from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.preprocessing.sequence import pad_sequences
from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.metrics import r2_score


vectors = KeyedVectors.load_word2vec_format(
    './data/vectors/GoogleNews-vectors-negative300.bin.gz',
    binary=True,
)


class Corpus:

    def __init__(self, pattern, skim=None):
        self.pattern = pattern
        self.skim = skim

    def lines(self):
        for path in glob(self.pattern):
            with open(path) as fh:
                for line in fh:
                    yield line.strip()

    def abstracts(self):
        lines = self.lines()
        if self.skim:
            lines = islice(lines, self.skim)
        for line in tqdm(lines, total=self.skim):
            raw = ujson.loads(line)
            yield Abstract.from_raw(raw)

    def xy(self):
        for abstract in self.abstracts():
            yield from abstract.xy()


class Abstract:

    @classmethod
    def from_raw(cls, raw):
        return cls([Sentence(s['token']) for s in raw['sentences']])

    def __init__(self, sentences):
        self.sentences = sentences

    def xy(self):
        for i, sent in enumerate(self.sentences):
            x = sent.token_vectors()
            y = i / (len(self.sentences)-1)
            yield x, y


class Sentence:

    def __init__(self, tokens):
        self.tokens = tokens

    def token_vectors(self):
        return [vectors[t] for t in self.tokens if t in vectors]


train = Corpus('./data/train.json/*.json', 100)
train_x, train_y = zip(*train.xy())
train_x = pad_sequences(train_x, 50, padding='post', dtype=float)
train_y = list(train_y)

model = Sequential()
model.add(LSTM(128, input_shape=train_x[0].shape))
model.add(Dense(1, kernel_initializer='normal'))
model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(train_x, train_y, batch_size=10)

dev = Corpus('./data/dev.json/*.json', 10)
dev_x, dev_y = zip(*train.xy())
dev_x = pad_sequences(dev_x, 50, padding='post', dtype=float)
dev_y = list(dev_y)

print(r2_score(dev_y, model.predict(dev_x)))

correct = Counter()
total = Counter()

for ab in dev.abstracts():

    x, _ = zip(*ab.xy())
    x = pad_sequences(x, 50, padding='post', dtype=float)

    preds = model.predict(x)
    order = list(preds[:,0].argsort().argsort())

    if sorted(order) == order:
        correct[len(order)] += 1

    total[len(order)] += 1


for slen in sorted(correct.keys()):
    print(slen, correct[slen] / total[slen])

print(sum(correct.values()) / sum(total.values()))
