import random
import numpy as np
import pandas as pd
import os
import gc
import re
from torch import nn
from keras.preprocessing.text import Tokenizer

from keras.layers import Input, Dense, Embedding, SpatialDropout1D, concatenate
from keras.layers import CuDNNLSTM, Bidirectional, GlobalMaxPooling1D, GlobalAveragePooling1D, CuDNNGRU

from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.losses import binary_crossentropy
from keras import backend as K
from tqdm import tqdm

from fastai.train import Learner
from fastai.train import DataBunch
from fastai.callbacks import TrainingPhase, GeneralScheduler
import torch
from torch.utils import data
from torch.nn import functional as F
import sys

package_dir = "../input/ppbert/pytorch-pretrained-bert/pytorch-pretrained-BERT"
sys.path.append(package_dir)
from pytorch_pretrained_bert import BertTokenizer, BertForSequenceClassification, BertAdam
from pytorch_pretrained_bert import BertConfig


EMB_MAX_FEAT = 300
MAX_LEN = 220
max_features = 400000
batch_size = 512
NUM_EPOCHS = 4
LSTM_UNITS = 128
DENSE_HIDDEN_UNITS = 512
NUM_MODELS = 1

EMB_PATHS = [
    '../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec',
    '../input/glove840b300dtxt/glove.840B.300d.txt'
]

JIGSAW_PATH = '../input/jigsaw-unintended-bias-in-toxicity-classification/'
print(os.listdir("../input/fasttext-crawl-300d-2m/"), os.listdir("../input/glove840b300dtxt/"), )


def C_trans_to_E():
    E_pun = 'BEFIKLMOQSTWZBCFGJKLMPVWXZCILOABCDFGHJKLMNOPQRSUVWXYABCDEHIJKLMOPRTWYBJKMVXZ012345678901234567FWY,;.!?:$ []()<>""\'\'AAAAAAAABBCCDDDDEEEEEEEEFFGGGHHHHIIIIIIJKLLLMNNNNNOOOOOOOPPRRRSSSSTTTTTTUUUUVWWYYYZPABBDEGHIJKLMNOPRTUWabdegkmnNoptuwvhjrwyxylsx\'\'    ' + 'ABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZ'
    C_pun = '𝗕𝗘𝗙𝗜𝗞𝗟𝗠𝗢𝗤𝗦𝗧𝗪𝗭𝗯𝗰𝗳𝗴𝗷𝗸𝗹𝗺𝗽𝘃𝘄𝘅𝘇𝘊𝘐𝘓𝘖𝘢𝘣𝘤𝘥𝘧𝘨𝘩𝘫𝘬𝘭𝘮𝘯𝘰𝘱𝘲𝘳𝘴𝘶𝘷𝘸𝘹𝘺𝘼𝘽𝘾𝘿𝙀𝙃𝙄𝙅𝙆𝙇𝙈𝙊𝙋𝙍𝙏𝙒𝙔𝙗𝙟𝙠𝙢𝙫𝙭𝙯𝟎𝟏𝟐𝟑𝟒𝟓𝟔𝟕𝟖𝟗𝟬𝟭𝟮𝟯𝟰𝟱𝟲𝟳🇫🇼🇾，；。！？：￥—【】（）《》“”‘’𝙖𝗮äаāàáᴀᴃʙᴄ𝙘ᴅ𝙙𝗱ᴆ𝙚𝖊𝘦𝗲éèêᴇғ𝙛ʛɢ𝙜ʜн𝗵𝙝í𝗶𝘪𝙞ïɪᴊᴋ𝙡ʟᴌᴍᴎ𝙣𝗻ñ𝖓ᴏо𝖔𝗼𝙤öóᴘ𝙥ʀ𝙧𝗿𝘀ѕś𝙨ᴛт𝖙𝘵𝘁𝙩ū𝙪𝘂ᴜᴠᴡ𝙬ʏ𝙮𝘆ᴢᴩᴬᴮᴯᴰᴱᴳᴴᴵᴶᴷᴸᴹᴺɴᴼᴾᴿᵀᵁᵂᵃᵇᵈᵉᵍᵏᵐᵑᵒᵖᵗᵘᵚᵛʰʲʳʷʸ˟ˠˡˢˣ´`…-_•' + '𝐀𝐁𝐂𝐃𝐄𝐅𝐆𝐇𝐈𝐉𝐊𝐋𝐌𝐍𝐎𝐏𝐐𝐑𝐒𝐓𝐔𝐕𝐖𝐗𝐘𝐙𝐚𝐛𝐜𝐝𝐞𝐟𝐠𝐡𝐢𝐣𝐤𝐥𝐦𝐧𝐨𝐩𝐪𝐫𝐬𝐭𝐮𝐯𝐰𝐱𝐲𝐳𝐴𝐵𝐶𝐷𝐸𝐹𝐺𝐻𝐼𝐽𝐾𝐿𝑀𝑁𝑂𝑃𝑄𝑅𝑆𝑇𝑈𝑉𝑊𝑋𝑌𝑍𝑎𝑏𝑐𝑑𝑒𝑓𝑔𝑖𝑗𝑘𝑙𝑚𝑛𝑜𝑝𝑞𝑟𝑠𝑡𝑢𝑣𝑤𝑥𝑦𝑧𝑨𝑩𝑪𝑫𝑬𝑭𝑮𝑯𝑰𝑱𝑲𝑳𝑴𝑵𝑶𝑷𝑸𝑹𝑺𝑻𝑼𝑽𝑾𝑿𝒀𝒁𝒂𝒃𝒄𝒅𝒆𝒇𝒈𝒉𝒊𝒋𝒌𝒍𝒎𝒏𝒐𝒑𝒒𝒓𝒔𝒕𝒖𝒗𝒘𝒙𝒚𝒛'
    table = {ord(f): ord(t) for f, t in zip(C_pun, E_pun)}
    return table


def char_to_word(string):
    char_list = string.split()
    new_word_list = []
    new_word = ""
    for i in range(len(char_list)):
        if len(char_list[i]) == 1 and char_list[i] in list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"):
            new_word += char_list[i]
        else:
            if len(new_word) > 3:
                new_word_list.append(new_word)
            new_word = ""
    
    if len(new_word_list) > 0:
        return new_word_list
    else:
        return np.nan


def build_doc_vocab(texts):
    sentences = texts.apply(lambda x: set(x.split())).values
    vocab = {}
    for sentence in sentences:
        for word in sentence:
            try:
                vocab[word] += 1
            except KeyError:
                vocab[word] = 1
    return vocab


def build_vocab(texts):
    sentences = texts.apply(lambda x: x.split()).values
    vocab = {}
    for sentence in sentences:
        for word in sentence:
            try:
                vocab[word] += 1
            except KeyError:
                vocab[word] = 1
    return vocab


def word_split(key, ngram, embeddings_index, vocab1):
    new_word_list = []
    length = len(key)
    if ngram == 2:
        for i in range(1, length - 1):
            if key[:i] in vocab1 and key[i:] in vocab1:
                new_word_list.append(" ".join([key[:i], key[i:]]))
    elif ngram == 3:
        for i in range(1, length - 2):
            for j in range(1, length - i - 1):
                if key[:i] in vocab1 and key[i:i + j] in vocab1 and key[i + j:] in vocab1:
                    new_word_list.append(" ".join([key[:i], key[i:i + j], key[i + j:]]))
    
    elif ngram == 4:
        for i in range(1, length - 3):
            for j in range(1, length - i - 2):
                for k in range(1, length - i - j - 1):
                    if key[:i] in vocab1 and key[i:i + j] in vocab1 and key[i + j:i + j + k] in vocab1 and key[
                                                                                                           i + j + k:] in vocab1:
                        new_word_list.append(" ".join([key[:i], key[i:i + j], key[i + j:i + j + k], key[i + j + k:]]))
    if len(new_word_list) > 0:
        return new_word_list
    else:
        return None


def build_vocab_ngram(texts, ngram, vocab1):
    sentences = texts.apply(lambda x: x.split()).values
    vocab = {}
    for sentence in sentences:
        if len(sentence) > ngram - 1:
            for i in range(len(sentence) - 1):
                if len([word for word in sentence[i:i + ngram] if word in vocab1]) > 0:
                    try:
                        vocab[" ".join(sentence[i:i + ngram])] += 1
                    except KeyError:
                        vocab[" ".join(sentence[i:i + ngram])] = 1
    return vocab


def long_word_dict(comment_text, embeddings_index, ngram):
    print("longword,start")
    vocab = build_vocab(comment_text)
    vocab = pd.Series(vocab).reset_index()
    vocab.columns = ["key", "cnt"]
    vocab1 = vocab[(vocab.cnt > 10) & (vocab.key.isin(embeddings_index))]
    vocab1 = vocab1.set_index("key").cnt.to_dict()
    vocab = vocab[
        (~vocab.key.isin(embeddings_index)) & (~vocab.key.str.contains("[^a-z0-9]")) & (vocab.key.str.len() > 5)]
    vocab_ngram = build_vocab_ngram(comment_text, ngram, vocab1)
    vocab["new_word_list"] = vocab.key.map(lambda x: word_split(x, ngram, embeddings_index, vocab1))
    
    def method_key(word):
        if word in vocab_ngram:
            return vocab_ngram[word]
        else:
            return 0
    
    vocab = vocab[vocab.new_word_list.notnull()]
    vocab["new_word"] = vocab["new_word_list"].map(lambda x: max(x, key=method_key))
    vocab = vocab.drop("new_word_list", axis=1)
    vocab["new_word_cnt"] = vocab["new_word"].map(method_key)
    vocab = vocab[vocab.new_word_cnt > 1]
    print("longword_dict_len", vocab.shape[0])
    return vocab.set_index("key").new_word.to_dict()


def long_word_dict_new(df, embeddings_index):
    print("longword,start")
    vocab = build_vocab(df.comment_text.str.lower())
    vocab = pd.Series(vocab).reset_index()
    vocab.columns = ["key", "cnt"]
    vocab1 = vocab[(vocab.cnt > 10) & (vocab.key.isin(embeddings_index))]
    vocab1 = vocab1.set_index("key").cnt.to_dict()
    vocab = vocab[
        (~vocab.key.isin(embeddings_index)) & (~vocab.key.str.contains("[^a-z0-9]")) & (vocab.key.str.len() > 5)]
    for ngram in range(2, 5):
        print(ngram, "start")
        vocab_ngram = build_vocab_ngram(df.comment_text[df.target > 0].str.lower(), ngram, vocab1)
        vocab["new_word_list{}".format(ngram)] = vocab.key.map(lambda x: word_split(x, ngram, embeddings_index, vocab1))
        
        def method_key(word):
            if word in vocab_ngram:
                return vocab_ngram[word]
            else:
                return 0
        
        vocab["new_word{}".format(ngram)] = vocab["new_word_list{}".format(ngram)].map(
            lambda x: max(x, key=method_key) if type(x) == list else None)
        vocab = vocab.drop("new_word_list{}".format(ngram), axis=1)
        vocab["new_word_cnt{}".format(ngram)] = vocab["new_word{}".format(ngram)].map(method_key)
        del vocab_ngram
        gc.collect()
    vocab = vocab[(vocab[["new_word_cnt2", "new_word_cnt3", "new_word_cnt4"]] > 1).any(1)]
    vocab["new_word_index"] = vocab[["new_word_cnt2", "new_word_cnt3", "new_word_cnt4"]].apply(
        lambda x: "new_word" + x.idxmax()[-1], axis=1)
    vocab["new_word"] = vocab.apply(lambda x: x[x.new_word_index], axis=1)
    return vocab.set_index("key").new_word.to_dict()


def word_regex(series, embeddings_index):
    string = series.comment_text
    for word in series.char_word:
        if word.lower() in embeddings_index:
            regex = ""
            for char_index in range(len(word)):
                regex += word[char_index]
                if char_index < len(word) - 1:
                    regex += "[^a-zA-Z]+"
            string = re.sub(regex, word.lower(), string)
    return string


def clean_punct(df, punct):
    for char in punct:
        regex = "\\" + char
        new_char = " " + char + " "
        df["comment_text"] = df.comment_text.str.replace(regex, new_char)
    return df


def long_word(key, emb_idx):
    length = len(key)
    for i in range(2, max(10, int(length / 2))):
        if length % i == 0 and key == "".join((list(key[:i]) * int(length / i))) and key[:i] in emb_idx:
            return key[:i], emb_idx[key[:i]]
    return key, None


def word_replace(text, replace_dict):
    text = ' '.join(
        [replace_dict[t] if t in replace_dict else replace_dict[t.lower()] if t.lower() in replace_dict else t for t in
         text.split()])
    return text


def text_clean(text):
    contraction_patterns = [(r'(\w+)\'ll', '\g<1> will'),
                            (r'(\w+)n\'t', '\g<1> not'),
                            (r'(\w+)\'ve', '\g<1> have'), (r'(\w+)\'s', '\g<1> is'),
                            (r'(\w+)\'re', '\g<1> are'), (r'(\w+)\'d', '\g<1> would')]
    patterns = [(re.compile(regex), repl) for (regex, repl) in contraction_patterns]
    for (pattern, repl) in patterns:
        (text, count) = re.subn(pattern, repl, text)
    return text


def build_char_count(texts):
    sentences = texts.apply(list).values
    vocab = {}
    for sentence in sentences:
        for char in sentence:
            try:
                vocab[char] += 1
            except KeyError:
                vocab[char] = 1
    return vocab


def char_clean(text, table):
    return text.translate(table)


def special_char_dict(df, embeddings_index):
    emb_iddx = set()
    for word in embeddings_index:
        emb_iddx = emb_iddx | set(list(word))
    vocab = build_char_count(df.comment_text)
    char_dict = {ord(char): "" for char in vocab.keys() if char not in emb_iddx and char not in list("\r\n\t ")}
    return char_dict


def multiple_replace(text, adict):
    rx = re.compile('|'.join(map(re.escape, adict)))
    
    def one_xlat(match):
        return adict[match.group(0)]
    
    return rx.sub(one_xlat, text)


def word_clean(word_list, vocab):
    word_dict = {}
    for word in word_list:
        regex = re.sub("\^|\$|\*|\@|\#", ".{0,1}", word)
        vocabx = vocab[vocab.index.map(lambda x: True if re.fullmatch(regex, x) else False)]
        if len(vocabx) > 0:
            word_dict[word] = vocabx.idxmax()
    return word_dict


def char_spell_check(word):
    char_list = list(word.lower())
    count = 0
    word_check = False
    char_set = []
    for char_index in range(len(char_list) - 1):
        if char_list[char_index] == char_list[char_index + 1]:
            count += 1
        else:
            if count > 2:
                char_set.append(char_list[char_index])
            if count > 3:
                word_check = True
            count = 0
    if count > 2:
        char_set.append(char_list[-1])
    if count > 3:
        word_check = True
    if word_check:
        return set(char_set)
    else:
        return np.nan


def spell_dict_create(vocab1, vocab):
    spell_dict = {}
    for index in vocab1.index:
        word = vocab1.word[index]
        regex = word.lower()
        for char in vocab1.char_set[index]:
            regex = re.sub("{}+".format(char), "{}+".format(char), regex)
        vocabx = vocab[vocab.index.map(lambda x: True if re.fullmatch(regex, x) else False)]
        if len(vocabx) > 0:
            spell_dict[word] = vocabx.idxmax()
    return spell_dict


def build_vocab_ngram_new(texts, ngram):
    sentences = texts.apply(lambda x: x.split()).values
    vocab = {}
    for sentence in sentences:
        if len(sentence) > ngram - 1:
            for i in range(len(sentence) - 1):
                try:
                    vocab[" ".join(sentence[i:i + ngram])] += 1
                except KeyError:
                    vocab[" ".join(sentence[i:i + ngram])] = 1
    return vocab


def word_merge_dict(df, embeddings_index):
    vocabp = build_vocab_ngram_new(df.comment_text, 2)
    vocabp = pd.Series(vocabp).reset_index()
    vocabp.columns = ["word2", "cnt"]
    vocabp = vocabp[~vocabp.word2.str.contains("[^a-zA-Z ]")]
    vocabp2 = vocabp.word2.str.split(" ", expand=True)
    vocabp2.columns = ["word0", "word1"]
    vocabp3 = (vocabp2.word0.str.lower().isin(embeddings_index)) & (vocabp2.word1.str.lower().isin(embeddings_index))
    vocabp4 = vocabp[~vocabp3]
    vocabp4["word3"] = vocabp4.word2.str.replace(" ", "")
    vocabp5 = vocabp4[vocabp4.word3.str.lower().isin(embeddings_index)]
    word_dict = vocabp5.set_index("word2").word3.to_dict()
    word_list = vocabp[(vocabp.word2.isin(word_dict.keys())) & (vocabp.cnt > 1)].word2.values
    return {key: word_dict[key] for key in word_list}


def convert_lines(example, max_seq_length, tokenizer):
    max_seq_length -= 2
    all_tokens = []
    longer = 0
    for text in example:
        tokens_a = tokenizer.tokenize(text)
        if len(tokens_a) > max_seq_length:
            tokens_a = tokens_a[:max_seq_length]
            longer += 1
        one_token = tokenizer.convert_tokens_to_ids(["[CLS]"] + tokens_a + ["[SEP]"]) + [0] * (
            max_seq_length - len(tokens_a))
        all_tokens.append(one_token)
    return np.array(all_tokens)


def bert_predict(test_df):
    BERT_MODEL_PATH = '../input/bert-pretrained-models/cased_l-12_h-768_a-12/cased_L-12_H-768_A-12/'
    bert_config = BertConfig('../input/casedmodel4/bert_config.json')
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH, cache_dir=None, do_lower_case=True)
    
    device = torch.device('cuda')
    model = BertForSequenceClassification(bert_config, num_labels=8)
    model.load_state_dict(torch.load("../input/casedmodel4/bert_pytorch.bin"))
    model.to(device)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    X_test = convert_lines(test_df["comment_text"].fillna("DUMMY_VALUE"), 220, tokenizer)
    test_preds = np.zeros((len(X_test)))
    test = torch.utils.data.TensorDataset(torch.tensor(X_test, dtype=torch.long))
    test_loader = torch.utils.data.DataLoader(test, batch_size=512, shuffle=False)
    for i, (x_batch,) in enumerate(test_loader):
        pred = model(x_batch.to(device), attention_mask=(x_batch > 0).to(device), labels=None)
        test_preds[i * 512:(i + 1) * 512] = pred[:, 0].detach().cpu().squeeze().numpy()
    
    test_pred = torch.sigmoid(torch.tensor(test_preds)).numpy().ravel()
    return test_pred


def load_data_and_clean(debug=False):
    embeddings_index = load_embeddings1(EMB_PATHS)
    train = pd.read_csv("../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv")
    test = pd.read_csv("../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv")
    if debug:
        train = train.iloc[:10000]
        test = test.iloc[:10000]
    df = pd.concat([train, test], ignore_index=True)[["comment_text", "target"]]
    del train, test
    gc.collect()
    df["comment_text"] = df.comment_text.fillna("")
    translate_dict = C_trans_to_E()
    df["comment_text"] = df.comment_text.map(lambda x: x.translate(translate_dict))
    df["comment_text"] = df.comment_text.str.replace("·", "").str.replace("\xad", "")
    df["comment_text"] = df.comment_text.replace("(U|u)\.(S|s)\.(A|a)", "USA")
    df["comment_text"] = df.comment_text.map(lambda x: re.sub("((ht|f)tps?):\/\/[^\s]*", "http ", x))
    df["comment_text"] = df.comment_text.map(lambda x: re.sub("[\u0800-\u4e00\uAC00-\uD7A3\u4E00-\u9FA5]", "", x))
    df = clean_punct(df, "?!.,()/+:;<=>[]{|}%&–~\"")
    df["char_word"] = df.comment_text.map(char_to_word)
    df.comment_text[df.char_word.notnull()] = df[df.char_word.notnull()].apply(
        lambda x: word_regex(x, embeddings_index), axis=1)
    df = df.drop(["char_word"], axis=1)
    replace_dict = {"'i'm": "i am", "'you're": "you are", "ain't": "is not", "aren't": "are not",
                    "arn't": "are not",
                    "c'mon": "common",
                    "could'nt": "could not", "could've": "could have", "couldn't": "could not",
                    "did'nt": "did not",
                    "din't": "did not",
                    "bullshet": "bullshit",
                    "colour": "color", "centre": "center", "favourite": "favorite",
                    "travelling": "traveling", "counselling": "counseling", "theatre": "theater",
                    "cancelled": "canceled", "labour": "labor", "organisation": "organization",
                    "narcisist": "narcissist", "qouta": "quota", "whst": "what",
                    "demonetisation": "demonetization",
                    "stooooooooooooooooooooopid": "stupid", "stoooooooooooopid": "stupid",
                    "stooooooopid": "stupid",
                    "stoooooopid": "stupid",
                    "doens't": "do not", "dont't": "do not", "dosen't": "do not",
                    "dosn't": "does not",
                    "gov't": "government",
                    "hadn't": "had not", "hasn't": "has not", "havn't": "have not",
                    "he'd": "he would",
                    "he'd've": "he would have",
                    "he'll": "he will", "here's": "here is", "how'd": "how did",
                    "how'd'y": "how do you",
                    "how'll": "how will",
                    "how's": "how is", "i'am": "i am", "i'd": "i would", "i'd've": "i would have",
                    "i'l": "i will",
                    "i'll": "i will",
                    "i'll've": "i will have", "i'm": "i am", "i'ma": "i am", "i'v": "i have",
                    "i've": "i have",
                    "is'nt": "is not",
                    "it'd": "it would", "it'd've": "it would have", "it'll": "it will",
                    "it'll've": "it will have",
                    "let's": "let us",
                    "ma'am": "madam", "mayn't": "may not", "might've": "might have",
                    "mightn't": "might not",
                    "mightn't've": "might not have", "must've": "must have", "mustn't": "must not",
                    "mustn't've": "must not have",
                    "needn't": "need not", "needn't've": "need not have",
                    "o'clock": "of the clock",
                    "oughtn't": "ought not",
                    "oughtn't've": "ought not have", "sha'n't": "shall not", "shan't": "shall not",
                    "shan't've": "shall not have",
                    "she'd": "she would", "she'd've": "she would have", "she'll": "she will",
                    "she'll've": "she will have",
                    "shoudn't": "should not", "should've": "should have",
                    "shouldn't": "should not",
                    "shouldn't've": "should not have",
                    "so's": "so as", "so've": "so have", "that'd": "that would",
                    "that'd've": "that would have",
                    "that'll": "that will",
                    "there'd": "there would", "there'd've": "there would have",
                    "there'll": "there will",
                    "there're": "there are",
                    "they'd": "they would", "they'd've": "they would have", "they'll": "they will",
                    "they'll've": "they will have",
                    "they've": "they have", "this'll": "this will", "this's": "this is",
                    "to've": "to have",
                    "wan't": "want",
                    "was'nt": "was not", "wasn't": "was not", "we'd": "we would",
                    "we'd've": "we would have",
                    "we'll": "we will",
                    "we'll've": "we will have", "we've": "we have", "weren't": "were not",
                    "what'll": "what will",
                    "what'll've": "what will have", "what're": "what are", "what's": "what is",
                    "what've": "what have",
                    "when's": "when is", "when've": "when have", "where'd": "where did",
                    "where's": "where is",
                    "where've": "where have",
                    "who'd": "who would", "who'll": "who will", "who'll've": "who will have",
                    "who're": "who are",
                    "who's": "who is",
                    "who've": "who have", "why'd": "why did", "why's": "why is",
                    "why've": "why have",
                    "will've": "will have",
                    "won't've": "will not have", "would've": "would have", "wouldn't": "would not",
                    "wouldn't've": "would not have",
                    "y'all": "you all", "y'all'd": "you all would",
                    "y'all'd've": "you all would have",
                    "y'all're": "you all are",
                    "y'all've": "you all have", "y'know": "you know", "ya'll": "you all",
                    "you'd": "you would",
                    "you'd've": "you would have", "you'll've": "you will have", "you'r": "you are",
                    "you've": "you have",
                    "your'e": "you are", "your're": "you are", "motherf*cking": "mother fucking"}
    vocab0 = pd.Series(build_doc_vocab(df[df.target == 0].comment_text))
    vocab1 = pd.Series(build_doc_vocab(df[df.target > 0].comment_text))
    vocab = vocab1[(~vocab1.index.isin(vocab0[vocab0 > 25].index)) & (vocab1 > 20) & (
        vocab1.index.str.lower().isin(embeddings_index)) & (~vocab1.index.str.contains("[^A-Za-z]")) & (
                           vocab1.index.str.len() > 3)]
    vocab_all = pd.Series(build_doc_vocab(df.comment_text))
    word_list = vocab_all[(~vocab_all.index.str.contains("[^A-Za-z\^\$\*\@\#]")) & (
        vocab_all.index.str.contains("[A-Za-z]")) & (vocab_all.index.str.contains("[\^\$\*\@\#]"))].index.values
    word_list = [i for i in word_list if len(re.sub("[^a-zA-Z]", "", i)) / len(i) >= 0.5]
    word_dict = word_clean(word_list, vocab)
    pd.Series(word_dict).to_json("word_dict.json")
    replace_dict = dict(replace_dict, **word_dict)
    df["comment_text"] = df.comment_text.map(lambda x: word_replace(x, replace_dict))
    df = clean_punct(df, "'*$^@#")
    del vocab, vocab0, vocab1
    gc.collect()
    longword_dict = long_word_dict_new(df, embeddings_index)
    pd.Series(longword_dict).to_json("longword_dict.json")
    vocab = pd.Series(build_doc_vocab(df.comment_text))
    vocab1 = vocab[~vocab.index.str.contains("[^a-zA-Z]")]
    vocab2 = vocab1[(vocab1 > 100) & (vocab1.index.isin(embeddings_index))]
    vocab1 = vocab1.reset_index()
    vocab1.columns = ["word", "cnt"]
    vocab1["char_set"] = vocab1.word.map(char_spell_check)
    spell_dict = spell_dict_create(
        vocab1[vocab1.char_set.notnull() & (~vocab1.word.str.lower().isin(embeddings_index))], vocab2)
    pd.Series(spell_dict).to_json("spell_dict.json")
    replace_dict = dict(replace_dict, **longword_dict, **spell_dict)
    del vocab, vocab1, vocab2
    gc.collect()
    df["comment_text"] = df.comment_text.map(lambda x: word_replace(x, replace_dict))
    pd.Series(replace_dict).to_json("replace_dict.json")
    if debug:
        pass
    else:
        word_merge = word_merge_dict(df, embeddings_index)
        pd.Series(word_merge).to_json("word_merge.json")
        def bool_series_create(x):
            
            if sum([1 if key in x else 0 for key in word_merge.keys()]) > 0:
                return True
            else:
                return False
        
        bool_series = df.comment_text.map(bool_series_create)
        df.comment_text[bool_series] = df.comment_text[bool_series].map(lambda x: multiple_replace(x, word_merge))
    gc.collect()
    df["length"] = df.comment_text.map(lambda x: len(x.split()))
    for length in range(220, 250):
        print(length, df[df.length > length].shape[0])
    train = pd.read_csv("../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv")
    test = pd.read_csv("../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv")
    if debug:
        train = train.iloc[:10000]
        test = test.iloc[:10000]
    train["comment_text"] = df[df.target.notnull()].comment_text.values
    test["comment_text"] = df[df.target.isnull()].comment_text.values
    del embeddings_index, df
    gc.collect()
    return train, test


def token_fit(train, test):
    identity_columns = ['asian', 'atheist', 'bisexual', 'black', 'buddhist', 'christian', 'female', 'heterosexual',
                        'hindu',
                        'homosexual_gay_or_lesbian', 'intellectual_or_learning_disability', 'jewish', 'latino', 'male',
                        'muslim', 'other_disability',
                        'other_gender', 'other_race_or_ethnicity', 'other_religion', 'other_sexual_orientation',
                        'physical_disability', 'psychiatric_or_mental_illness', 'transgender', 'white']
    # Overall
    weights = np.ones((len(train),)) / 4
    # Subgroup
    weights += (train[identity_columns].fillna(0).values >= 0.5).sum(axis=1).astype(bool).astype(np.int) / 4
    # Background Positive, Subgroup Negative
    weights += (((train['target'].values >= 0.5).astype(bool).astype(np.int) +
                 (train[identity_columns].fillna(0).values < 0.5).sum(axis=1).astype(bool).astype(np.int)) > 1).astype(
        bool).astype(np.int) / 4
    # Background Negative, Subgroup Positive
    weights += (((train['target'].values < 0.5).astype(bool).astype(np.int) +
                 (train[identity_columns].fillna(0).values >= 0.5).sum(axis=1).astype(bool).astype(np.int)) > 1).astype(
        bool).astype(np.int) / 4
    loss_weight = 1.0 / weights.mean()
    tok = Tokenizer(num_words=max_features, filters="", lower=False)
    tok.fit_on_texts(list(train.comment_text) + list(test.comment_text))
    x_train = tok.texts_to_sequences(train.comment_text.values)
    train_lengths = torch.from_numpy(np.array([len(x) for x in x_train]))
    x_train = torch.from_numpy(pad_sequences(x_train, maxlen=MAX_LEN))
    x_test = tok.texts_to_sequences(test.comment_text.values)
    test_lengths = torch.from_numpy(np.array([len(x) for x in x_test]))
    x_test = torch.from_numpy(pad_sequences(x_test, maxlen=MAX_LEN))
    y_train = np.vstack([(train['target'].values >= 0.5).astype(np.int), weights]).T
    y_train1 = train[['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']].values
    y_train2 = np.vstack([(train.target.values > i).astype(int) for i in [0, 0.25, 0.5, 0.75]])
    y_train2 = np.vstack([y_train2, weights]).T
    return x_train, x_test, y_train, y_train1, y_train2, tok, loss_weight, train_lengths, test_lengths


def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')


def load_embeddings1(path_list):
    emb_idx = []
    for path in path_list:
        for line in open(path):
            emb_idx.append(line.strip().split(" ")[0].lower())
    return list(set(emb_idx))


def load_embeddings(path):
    with open(path) as f:
        return dict(get_coefs(*line.strip().split(' ')) for line in f)


def build_embedding_matrix(tok, path):
    seed = 0
    all_know = []
    word_docs = tok.word_docs
    word_index = tok.word_index
    embeddings_index = load_embeddings(path)
    embedding_matrix = np.zeros((len(word_index) + 1, EMB_MAX_FEAT))
    vocabs = {key: word_docs[key] for key in word_docs.keys() if word_docs[key] > 0 and key in embeddings_index}
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    
    def edits1(word):
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [a + b[1:] for a, b in splits if b]
        transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b) > 1]
        replaces = [a + c + b[1:] for a, b in splits for c in alphabet if b]
        inserts = [a + c + b for a, b in splits for c in alphabet]
        return set(deletes + transposes + replaces + inserts)
    
    def known_edits2(word):
        return set(e2 for e1 in edits1(word) for e2 in edits1(e1) if e2 in vocabs)
    
    def known(words):
        return set(w for w in words if w in vocabs)
    
    def correct(word):
        candidates = known([word]) or known(edits1(word)) or [word]
        return max(candidates, key=vocabs.get)
    
    for key, i in word_index.items():
        if i <= max_features:
            word_know = [key]
            word = key
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                word_know.extend([word, "start"])
                all_know.append(word_know)
                embedding_matrix[i] = embedding_vector
                continue
            word = key.lower()
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                word_know.extend([word, "lower"])
                all_know.append(word_know)
                embedding_matrix[i] = embedding_vector
                continue
            word = key.upper()
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                word_know.extend([word, "upper"])
                all_know.append(word_know)
                embedding_matrix[i] = embedding_vector
                continue
            word = key.capitalize()
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                word_know.extend([word, "capitalize"])
                all_know.append(word_know)
                embedding_matrix[i] = embedding_vector
                continue
            word = ps.stem(key)
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                word_know.extend([word, "ps_stem"])
                all_know.append(word_know)
                embedding_matrix[i] = embedding_vector
                continue
            word = lc.stem(key)
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                word_know.extend([word, "lc_stem"])
                all_know.append(word_know)
                embedding_matrix[i] = embedding_vector
                continue
            word = sb.stem(key)
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                word_know.extend([word, "sb_stem"])
                all_know.append(word_know)
                embedding_matrix[i] = embedding_vector
                continue
            
            if len(key) > 5:
                word, embedding_vector = long_word(key, embeddings_index)
                if embedding_vector is not None:
                    word_know.extend([word, "long_word"])
                    all_know.append(word_know)
                    embedding_matrix[i] = embedding_vector
                    continue
            word = correct(key)
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                word_know.extend([word, "correct"])
                all_know.append(word_know)
                embedding_matrix[i] = embedding_vector
                continue
            if re.search("[a-zA-Z]", key) and re.search("[^a-zA-Z]", key):
                key1 = re.sub("[^a-zA-Z]", "", key)
                word = key1
                embedding_vector = embeddings_index.get(word)
                if embedding_vector is not None:
                    word_know.extend([word, "clean_start"])
                    all_know.append(word_know)
                    embedding_matrix[i] = embedding_vector
                    continue
                word = key1.lower()
                embedding_vector = embeddings_index.get(word)
                if embedding_vector is not None:
                    word_know.extend([word, "clean_lower"])
                    all_know.append(word_know)
                    embedding_matrix[i] = embedding_vector
                    continue
                word = key1.upper()
                embedding_vector = embeddings_index.get(word)
                if embedding_vector is not None:
                    word_know.extend([word, "clean_upper"])
                    all_know.append(word_know)
                    embedding_matrix[i] = embedding_vector
                    continue
                word = key1.capitalize()
                embedding_vector = embeddings_index.get(word)
                if embedding_vector is not None:
                    word_know.extend([word, "clean_capitalize"])
                    all_know.append(word_know)
                    embedding_matrix[i] = embedding_vector
                    continue
                word = ps.stem(key1)
                embedding_vector = embeddings_index.get(word)
                if embedding_vector is not None:
                    word_know.extend([word, "clean_ps_stem"])
                    all_know.append(word_know)
                    embedding_matrix[i] = embedding_vector
                    continue
                word = lc.stem(key1)
                embedding_vector = embeddings_index.get(word)
                if embedding_vector is not None:
                    word_know.extend([word, "clean_lc_stem"])
                    all_know.append(word_know)
                    embedding_matrix[i] = embedding_vector
                    continue
                word = correct(key1)
                embedding_vector = embeddings_index.get(word)
                if embedding_vector is not None:
                    word_know.extend([word, "clean_correct"])
                    all_know.append(word_know)
                    embedding_matrix[i] = embedding_vector
                    continue
            embedding_matrix[i] -= 1
            word_know.extend(["unknow", "unknow"])
            all_know.append(word_know)
    pd.DataFrame(all_know, columns=["key", "word", "method"]).to_csv(path[-3:] + ".csv", index=False)
    del embeddings_index, all_know, vocabs
    gc.collect()
    return embedding_matrix.astype(np.float32)


def build_embeddings(tok):
    print("embedding")
    embedding_matrix = np.concatenate(
        [build_embedding_matrix(tok, f) for f in EMB_PATHS], axis=-1)
    return embedding_matrix


def build_model(embedding_matrix, num_aux_targets, loss_weight):
    '''
    credits go to: https://www.kaggle.com/thousandvoices/simple-lstm/
    '''
    words = Input(shape=(MAX_LEN,))
    x = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False)(words)
    x = SpatialDropout1D(0.3)(x)
    x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(x)
    x = Bidirectional(CuDNNGRU(LSTM_UNITS, return_sequences=True))(x)
    
    hidden = concatenate([GlobalMaxPooling1D()(x), GlobalAveragePooling1D()(x)])
    result = Dense(3, activation='sigmoid')(hidden)
    aux_result = Dense(num_aux_targets, activation='sigmoid')(hidden)
    
    model = Model(inputs=words, outputs=[result, aux_result])
    model.compile(loss=[custom_loss, 'binary_crossentropy'], loss_weights=[loss_weight, 1.0], optimizer='adam')
    
    return model


def submit(sub_preds, debug):
    submission = pd.read_csv(os.path.join(JIGSAW_PATH, 'sample_submission.csv'), index_col='id')
    if debug:
        submission = submission.iloc[:10000]
    submission['prediction'] = sub_preds
    submission.reset_index(drop=False, inplace=True)
    submission.to_csv('submission.csv', index=False)


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class SpatialDropout(nn.Dropout2d):
    def forward(self, x):
        x = x.unsqueeze(2)  # (N, T, 1, K)
        x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)
        x = super(SpatialDropout, self).forward(x)  # (N, K, 1, T), some features are masked
        x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)
        x = x.squeeze(2)  # (N, T, K)
        return x


class SequenceBucketCollator():
    def __init__(self, choose_length, sequence_index, length_index, label_index=None):
        self.choose_length = choose_length
        self.sequence_index = sequence_index
        self.length_index = length_index
        self.label_index = label_index
    
    def __call__(self, batch):
        batch = [torch.stack(x) for x in list(zip(*batch))]
        
        sequences = batch[self.sequence_index]
        lengths = batch[self.length_index]
        
        length = self.choose_length(lengths)
        mask = torch.arange(start=MAX_LEN, end=0, step=-1) < length
        padded_sequences = sequences[:, mask]
        
        batch[self.sequence_index] = padded_sequences
        
        if self.label_index is not None:
            return [x for i, x in enumerate(batch) if i != self.label_index], batch[self.label_index]
        
        return batch


class NeuralNet(nn.Module):
    def __init__(self, embedding_matrix, num_aux_targets, num_targets):
        super(NeuralNet, self).__init__()
        embed_size = embedding_matrix.shape[1]
        
        self.embedding = nn.Embedding(max_features, embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        self.embedding_dropout = SpatialDropout(0.3)
        
        self.lstm1 = nn.LSTM(embed_size, LSTM_UNITS, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(LSTM_UNITS * 2, LSTM_UNITS, bidirectional=True, batch_first=True)
        
        self.linear1 = nn.Linear(DENSE_HIDDEN_UNITS, DENSE_HIDDEN_UNITS)
        self.linear2 = nn.Linear(DENSE_HIDDEN_UNITS, DENSE_HIDDEN_UNITS)
        
        self.linear_out = nn.Linear(DENSE_HIDDEN_UNITS, num_targets)
        self.linear_aux_out = nn.Linear(DENSE_HIDDEN_UNITS, num_aux_targets)
    
    def forward(self, x, lengths=None):
        h_embedding = self.embedding(x.long())
        h_embedding = self.embedding_dropout(h_embedding)
        
        h_lstm1, _ = self.lstm1(h_embedding)
        h_lstm2, _ = self.lstm2(h_lstm1)
        
        # global average pooling
        avg_pool = torch.mean(h_lstm2, 1)
        # global max pooling
        max_pool, _ = torch.max(h_lstm2, 1)
        
        h_conc = torch.cat((max_pool, avg_pool), 1)
        h_conc_linear1 = F.relu(self.linear1(h_conc))
        h_conc_linear2 = F.relu(self.linear2(h_conc))
        
        hidden = h_conc + h_conc_linear1 + h_conc_linear2
        
        result = self.linear_out(hidden)
        aux_result = self.linear_aux_out(hidden)
        out = torch.cat([result, aux_result], 1)
        
        return out


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def train_model(learn, test, output_dim, lr=0.001,
                batch_size=512, n_epochs=4,
                enable_checkpoint_ensemble=True):
    all_test_preds = []
    sub=pd.read_csv(os.path.join(JIGSAW_PATH, 'sample_submission.csv'))
    checkpoint_weights = [2 ** epoch for epoch in range(n_epochs)]
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)
    n = len(learn.data.train_dl)
    phases = [(TrainingPhase(n).schedule_hp('lr', lr * (0.6 ** (i)))) for i in range(n_epochs)]
    sched = GeneralScheduler(learn, phases)
    learn.callbacks.append(sched)
    for epoch in range(n_epochs):
        learn.fit(1)
        test_preds = np.zeros((len(test), output_dim))
        for i, x_batch in enumerate(test_loader):
            X = x_batch[0].cuda()
            y_pred = sigmoid(learn.model(X).detach().cpu().numpy())
            test_preds[i * batch_size:(i + 1) * batch_size, :] = y_pred
        sub["pre"+str(epoch)]=test_preds[:,0]
        all_test_preds.append(test_preds)
    
    if enable_checkpoint_ensemble:
        test_preds = np.average(all_test_preds, weights=checkpoint_weights, axis=0)
    else:
        test_preds = all_test_preds[-1]
    sub.to_csv("subx.csv",index=False)
    return test_preds


def custom_loss(data, targets):
    ''' Define custom loss function for weighted BCE on 'target' column '''
    bce_loss_1 = nn.BCEWithLogitsLoss(weight=targets[:, 1:2])(data[:, :1], targets[:, :1])
    bce_loss_2 = nn.BCEWithLogitsLoss()(data[:, 1:], targets[:, 2:])
    return (bce_loss_1 * loss_weight) + bce_loss_2


def custom_loss1(data, targets):
    ''' Define custom loss function for weighted BCE on 'target' column '''
    bce_loss_1 = nn.BCEWithLogitsLoss(weight=targets[:, 4:5])(data[:, :4], targets[:, :4])
    bce_loss_2 = nn.BCEWithLogitsLoss()(data[:, 4:], targets[:, 5:])
    return (bce_loss_1 * loss_weight) + bce_loss_2


def train_(x_train, y_train, y_aux_train, x_test):
    
    y_train_torch = torch.tensor(np.hstack([y_train, y_aux_train]), dtype=torch.float32)
    test_dataset = data.TensorDataset(x_test, test_lengths)
    train_dataset = data.TensorDataset(x_train, train_lengths, y_train_torch)
    valid_dataset = data.Subset(train_dataset, indices=[0, 1])
    del x_train, x_test
    gc.collect()
    
    train_collator = SequenceBucketCollator(lambda lenghts: lenghts.max(),
                                            sequence_index=0,
                                            length_index=1,
                                            label_index=2)
    
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_collator)
    valid_loader = data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=train_collator)
    
    databunch = DataBunch(train_dl=train_loader, valid_dl=valid_loader, collate_fn=train_collator)
    
    del train_dataset, valid_dataset
    gc.collect()
    
    for model_idx in range(NUM_MODELS):
        all_test_preds = []
        print('Model ', model_idx)
        seed_everything(1234 + model_idx)
        model = NeuralNet(embedding_matrix, y_aux_train.shape[-1], y_train.shape[-1] - 1)
        if y_train.shape[-1] > 2:
            learn = Learner(databunch, model, loss_func=custom_loss1)
        else:
            learn = Learner(databunch, model, loss_func=custom_loss)
        test_preds = train_model(learn, test_dataset, output_dim=y_train.shape[-1] + y_aux_train.shape[-1] - 1)
        all_test_preds.append(test_preds)
    preds = np.mean(all_test_preds, axis=0)
    return preds


debug = False
train, test = load_data_and_clean(debug)
gc.collect()

x_train, x_test, y_train, y_train1, y_train2, tok, loss_weight, train_lengths, test_lengths = token_fit(train, test)
print("max_features", len(tok.word_index) + 1)
max_features = min(max_features, len(tok.word_index) + 1)
preds2 = bert_predict(test)
del train, test
gc.collect()

#preds0 = train_(x_train, y_train, y_train1, x_test)
sub_preds = preds2
submit(sub_preds, debug)
#submission = pd.read_csv(os.path.join(JIGSAW_PATH, 'sample_submission.csv'))
#if debug:
#    submission=submission.iloc[:10000]
#submission["pre0"] = preds0[:, 0]
#submission["pre1"] = preds2
#submission.to_csv("sub.csv", index=False)