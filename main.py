import torch
import torch.nn as nn
import torch.optim as optimizers
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tensorflow.keras.preprocessing.sequence import pad_sequences

import random
from tqdm.notebook import tqdm
import pickle
import numpy as np
import matplotlib.pyplot as plt

PATH = "."
data = []
num_files = int(input("Enter the number of pickle files: "))
for i in range(num_files):
    file_name = input(f"Enter the name of file{i+1}: ")
    # ファイルの読み込み
    data += pickle.load(open(PATH+"/text/"+file_name+".pickle", "rb"))

# 単語をidにする
class EncoderDecoder():
    # インスタンス生成時に呼び出され、インスタンスの初期化を行う
    def __init__(self):
        self.w2i = {}  # word_to_idの辞書
        self.i2w = {}  # id_to_wordの辞書

        # 予約語
        self.special_chars = ['<pad>', '<s>', '</s>', '<unk>']
        self.bos_char = self.special_chars[1]
        self.eos_char = self.special_chars[2]
        self.oov_char = self.special_chars[3]

    # インスタンスから呼び出される
    def __call__(self, sentence):
        return self.transform(sentence)

    # 辞書作成
    def fit(self, sentences):
        self._words = set()

        # 未知の単語の集合を作成する
        for sentence in sentences:
            self._words.update(sentence)

        # 予約語分ずらしてidを振る
        self.w2i = {w: (i + len(self.special_chars)) for i, w in enumerate(self._words)}

        # 予約語を辞書に追加する(<pad>:0, <s>:1, </s>:2, <unk>:3)
        for i, w in enumerate(self.special_chars):
            self.w2i[w] = i

        # word_to_idの辞書を用いてid_to_wordの辞書を作成する
        self.i2w = {i: w for w, i in self.w2i.items()}

    # 読み込んだデータをまとめてidに変換する
    def transform(self, sentences, bos=True, eos=True):
        output = []
        # 指定があれば始まりと終わりの記号を追加する
        for sentence in sentences:
            if bos:
                sentence = [self.bos_char] + sentence
            if eos:
                sentence = sentence + [self.eos_char]
            output.append(self.encode(sentence))

        return output

    # 1文ずつidにする
    def encode(self, sentence):
        output = []
        for w in sentence:
            if w not in self.w2i:
                idx = self.w2i[self.oov_char]
            else:
                idx = self.w2i[w]
            output.append(idx)

        return output

    # １文ずつ単語リストに直す
    def decode(self, sentence):
        return [self.i2w[id] for id in sentence]

en_de = EncoderDecoder()  # インスタンスの生成
en_de.fit(data)  # 辞書作成
data_id = en_de(data)  # __call__を呼び出す（データをidに変換）

# データとラベルを作成する
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data, max_length=50):
        self.data_num = len(data)

        # データを1つずつずらす
        self.x = [d[:-1] for d in data]
        self.y = [d[1:] for d in data]

        # パディングして合わせる長さ
        self.max_length = max_length

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        out_data = self.x[idx]
        out_label =  self.y[idx]

        # パディングして長さを合わせる
        out_data = pad_sequences([out_data], padding='post', maxlen=self.max_length)[0]
        out_label = pad_sequences([out_label], padding='post', maxlen=self.max_length)[0]

        # LongTensor型に変換する
        out_data = torch.LongTensor(out_data)
        out_label = torch.LongTensor(out_label)

        return out_data, out_label

dataset = MyDataset(data_id, max_length=50)

# 9:1にデータを分割
train_size = int(0.9 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# DataLoaderでバッチ単位にする
train_data_loader = DataLoader(train_dataset, batch_size=50, drop_last=True)
test_data_loader = DataLoader(test_dataset, batch_size=1, drop_last=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderRNN(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, batch_size=100, num_layers=1):
        super(EncoderRNN, self).__init__()
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout()
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True, num_layers=self.num_layers)

    # 隠れ状態の初期化
    def init_hidden(self):
        self.hidden_state = torch.zeros(self.num_layers, self.batch_size, self.hidden_dim, device=device)

    # 与えられた単語IDを単語分散表現にしてからGRUに渡す
    def forward(self, indices):
        embed = self.word_embeddings(indices)
        embed = self.dropout(embed)

        # 単語分散表現と隠れ状態をGRUに渡す
        # ３次元（batch, seq_len, input_size）
        _, self.hidden_state = self.gru(embed, self.hidden_state)

        return self.hidden_state

class DecoderRNN(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, batch_size=100, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout()

        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True, num_layers=self.num_layers)
        self.output = nn.Linear(hidden_dim, vocab_size)

    # 与えられた単語IDを単語分散表現にしてからGRUに渡し、Linear layerから出力
    def forward(self, indices, hidden):
        self.hidden_state = hidden
        embed = self.word_embeddings(indices)
        embed = self.dropout(embed)

        # 単語分散表現と隠れ状態をGRUに渡す
        # ３次元（batch, seq_len, input_size）
        gru_out, self.hidden_state = self.gru(embed, self.hidden_state)

        out = self.output(gru_out)  # Linear layerから出力

        return out, self.hidden_state


# 学習のための設定
EMBEDDING_DIM = HIDDEN_DIM = 256
VOCAB_SIZE = len(en_de.i2w)  # 扱う文字の数（辞書のサイズ）
BATCH_SIZE=50
encoder = EncoderRNN(EMBEDDING_DIM, HIDDEN_DIM, VOCAB_SIZE, batch_size=BATCH_SIZE).to(device)
decoder = DecoderRNN(EMBEDDING_DIM, HIDDEN_DIM, VOCAB_SIZE, batch_size=BATCH_SIZE).to(device)

# 損失関数
criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=0)

# 最適化
encoder_optimizer = optimizers.Adam(encoder.parameters(), lr=0.001, betas=(0.9, 0.999), amsgrad=True)
decoder_optimizer = optimizers.Adam(decoder.parameters(), lr=0.001, betas=(0.9, 0.999), amsgrad=True)

hist = {'train_loss': [], 'ppl':[]}
EPOCHS = 500

def compute_loss(label, pred):
    return criterion(pred, label)

def train_step(x, t):
    # 学習モードに入る（Dropoutを使うため）
    encoder.train()
    decoder.train()

    encoder.init_hidden()  # 隠れ状態の初期化
    encoder_hidden = encoder(x)  # Encoderの順伝搬

    decoder_hidden = encoder_hidden

    preds, _ = decoder(x, decoder_hidden)  # 予測

    # 教師データと予測の誤差を計算
    loss = compute_loss(t.view(-1), preds.view(-1, preds.size(-1)))

    # 勾配の初期化
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    loss.backward()  # 誤差逆伝播

    # パラメータ更新
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss, preds

for epoch in tqdm(range(EPOCHS)):
    train_loss = 0.
    loss_count = 0

    for (x, t) in train_data_loader:
        x, t = x.to(device), t.to(device)
        loss, _ = train_step(x, t)
        train_loss += loss.item()
        loss_count += 1

    # perplexity
    ppl = np.exp(train_loss / loss_count)
    train_loss /= len(train_data_loader)

    hist["train_loss"].append(train_loss)
    hist["ppl"].append(ppl)

    # 20epochごとに表示・保存
    if (epoch+1) % 20 == 0:
        print('-' * 20)
        print('epoch: {}'.format(epoch+1))
        print('train_loss: {:.3f}, ppl: {:.3f}'.format(
            train_loss, ppl
        ))

MODEL_PATH =PATH+"/embedding{}_v{}.model".format(EMBEDDING_DIM, EPOCHS)
torch.save({
    'encoder_model': encoder.state_dict(),
    'decoder_model': decoder.state_dict()
}, MODEL_PATH)

# 誤差の可視化
train_loss = hist['train_loss']
fig = plt.figure(figsize=(10, 5))
plt.plot(range(len(train_loss)), train_loss, linewidth=1, label='train_loss')
plt.xlabel('epochs')
plt.ylabel('train_loss')
plt.legend()
plt.savefig(PATH+'/img/loss.jpg')
plt.show()

# 予測
def generate_sentence(model_path, embedding_dim, hidden_dim, vocab_size, batch_size=1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = EncoderRNN(embedding_dim, hidden_dim, vocab_size, batch_size).to(device)
    decoder = DecoderRNN(embedding_dim, hidden_dim, vocab_size, batch_size).to(device)
    checkpoint = torch.load(model_path)
    encoder.load_state_dict(checkpoint["encoder_model"])
    decoder.load_state_dict(checkpoint["decoder_model"])

    with torch.no_grad():
        for (x, t) in test_data_loader:
            x, t = x.to(device), t.to(device)
            sentence = []
            decoder_input_idx = en_de.encode(["<s>"])
            decoder_input = torch.tensor([decoder_input_idx], device=device)

            encoder.eval()
            decoder.eval()

            encoder.init_hidden()
            encoder_hidden = encoder(x)
            decoder_hidden = encoder_hidden

            for _ in range(50):
                out, decoder_hidden = decoder(decoder_input, decoder_hidden)
                out = torch.squeeze(out)
                probs = F.softmax(out, dim=0)
                p = probs.cpu().detach().numpy()
                morpheme = en_de.i2w[np.random.choice(len(p), p=p)]

                decoder_input_idx = en_de.encode([morpheme])
                decoder_input = torch.tensor([decoder_input_idx], device=device)

                sentence.append(morpheme)

                if morpheme in ["</s>", "<pad>"]:
                    break

            print("".join(sentence).replace("</s>", "。"))
            print('-' * 50)

generate_sentence(MODEL_PATH, EMBEDDING_DIM, HIDDEN_DIM, VOCAB_SIZE)
