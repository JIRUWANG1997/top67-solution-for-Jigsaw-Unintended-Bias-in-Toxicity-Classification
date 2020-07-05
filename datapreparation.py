import pandas as pd
import gc
import re
import json
import numpy as np
EMB_PATHS = [
    '../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec',
    '../input/glove840b300dtxt/glove.840B.300d.txt'
]
class datapreparation():
    def __init__(self,debug):
        self.debug = debug
        self.train = None
        self.test = None

    def C_trans_to_E(self):
        E_pun = 'BEFIKLMOQSTWZBCFGJKLMPVWXZCILOABCDFGHJKLMNOPQRSUVWXYABCDEHIJKLMOPRTWYBJKMVXZ012345678901234567FWY,;.!?:$ []()<>""\'\'AAAAAAAABBCCDDDDEEEEEEEEFFGGGHHHHIIIIIIJKLLLMNNNNNOOOOOOOPPRRRSSSSTTTTTTUUUUVWWYYYZPABBDEGHIJKLMNOPRTUWabdegkmnNoptuwvhjrwyxylsx\'\'    ' + 'ABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZ'
        C_pun = '𝗕𝗘𝗙𝗜𝗞𝗟𝗠𝗢𝗤𝗦𝗧𝗪𝗭𝗯𝗰𝗳𝗴𝗷𝗸𝗹𝗺𝗽𝘃𝘄𝘅𝘇𝘊𝘐𝘓𝘖𝘢𝘣𝘤𝘥𝘧𝘨𝘩𝘫𝘬𝘭𝘮𝘯𝘰𝘱𝘲𝘳𝘴𝘶𝘷𝘸𝘹𝘺𝘼𝘽𝘾𝘿𝙀𝙃𝙄𝙅𝙆𝙇𝙈𝙊𝙋𝙍𝙏𝙒𝙔𝙗𝙟𝙠𝙢𝙫𝙭𝙯𝟎𝟏𝟐𝟑𝟒𝟓𝟔𝟕𝟖𝟗𝟬𝟭𝟮𝟯𝟰𝟱𝟲𝟳🇫🇼🇾，；。！？：￥—【】（）《》“”‘’𝙖𝗮äаāàáᴀᴃʙᴄ𝙘ᴅ𝙙𝗱ᴆ𝙚𝖊𝘦𝗲éèêᴇғ𝙛ʛɢ𝙜ʜн𝗵𝙝í𝗶𝘪𝙞ïɪᴊᴋ𝙡ʟᴌᴍᴎ𝙣𝗻ñ𝖓ᴏо𝖔𝗼𝙤öóᴘ𝙥ʀ𝙧𝗿𝘀ѕś𝙨ᴛт𝖙𝘵𝘁𝙩ū𝙪𝘂ᴜᴠᴡ𝙬ʏ𝙮𝘆ᴢᴩᴬᴮᴯᴰᴱᴳᴴᴵᴶᴷᴸᴹᴺɴᴼᴾᴿᵀᵁᵂᵃᵇᵈᵉᵍᵏᵐᵑᵒᵖᵗᵘᵚᵛʰʲʳʷʸ˟ˠˡˢˣ´`…-_•' + '𝐀𝐁𝐂𝐃𝐄𝐅𝐆𝐇𝐈𝐉𝐊𝐋𝐌𝐍𝐎𝐏𝐐𝐑𝐒𝐓𝐔𝐕𝐖𝐗𝐘𝐙𝐚𝐛𝐜𝐝𝐞𝐟𝐠𝐡𝐢𝐣𝐤𝐥𝐦𝐧𝐨𝐩𝐪𝐫𝐬𝐭𝐮𝐯𝐰𝐱𝐲𝐳𝐴𝐵𝐶𝐷𝐸𝐹𝐺𝐻𝐼𝐽𝐾𝐿𝑀𝑁𝑂𝑃𝑄𝑅𝑆𝑇𝑈𝑉𝑊𝑋𝑌𝑍𝑎𝑏𝑐𝑑𝑒𝑓𝑔𝑖𝑗𝑘𝑙𝑚𝑛𝑜𝑝𝑞𝑟𝑠𝑡𝑢𝑣𝑤𝑥𝑦𝑧𝑨𝑩𝑪𝑫𝑬𝑭𝑮𝑯𝑰𝑱𝑲𝑳𝑴𝑵𝑶𝑷𝑸𝑹𝑺𝑻𝑼𝑽𝑾𝑿𝒀𝒁𝒂𝒃𝒄𝒅𝒆𝒇𝒈𝒉𝒊𝒋𝒌𝒍𝒎𝒏𝒐𝒑𝒒𝒓𝒔𝒕𝒖𝒗𝒘𝒙𝒚𝒛'
        table = {ord(f): ord(t) for f, t in zip(C_pun, E_pun)}
        return table

    def multiple_replace(self,text, adict):
        rx = re.compile('|'.join(map(re.escape, adict)))

        def one_xlat(match):
            return adict[match.group(0)]

        return rx.sub(one_xlat, text)
    def load_embeddings1(self,path_list):
        emb_idx = []
        for path in path_list:
            for line in open(path):
                emb_idx.append(line.strip().split(" ")[0].lower())
        return list(set(emb_idx))

    def char_to_word(self,string):
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

    def clean_punct(self,df, punct):
        for char in punct:
            regex = "\\" + char
            new_char = " " + char + " "
            df["comment_text"] = df.comment_text.str.replace(regex, new_char)
        return df
    def load_embeddings1(self,path_list):
        emb_idx = []
        for path in path_list:
            for line in open(path):
                emb_idx.append(line.strip().split(" ")[0].lower())
        return list(set(emb_idx))

    def word_regex(self,series, embeddings_index):
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

    def word_replace(self,text, replace_dict):
        text = ' '.join(
            [replace_dict[t] if t in replace_dict else replace_dict[t.lower()] if t.lower() in replace_dict else t for t
             in
             text.split()])
        return text

    def load_data_and_clean(self,debug=False):
        embeddings_index = self.load_embeddings1(EMB_PATHS)
        self.train = pd.read_csv("../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv")
        self.test = pd.read_csv("../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv")
        if debug:
            self.train = self.train.iloc[:10000]
            self.test = self.test.iloc[:10000]
        df = pd.concat([self.train, self.test], ignore_index=True)[["comment_text", "target"]]
        del self.train, self.test
        gc.collect()
        df["comment_text"] = df.comment_text.fillna("")
        translate_dict = self.C_trans_to_E()
        df["comment_text"] = df.comment_text.map(lambda x: x.translate(translate_dict))
        df["comment_text"] = df.comment_text.str.replace("·", "").str.replace("\xad", "")
        df["comment_text"] = df.comment_text.replace("(U|u)\.(S|s)\.(A|a)", "USA")
        df["comment_text"] = df.comment_text.map(lambda x: re.sub("((ht|f)tps?):\/\/[^\s]*", "http ", x))
        df["comment_text"] = df.comment_text.map(lambda x: re.sub("[\u0800-\u4e00\uAC00-\uD7A3\u4E00-\u9FA5]", "", x))
        df = self.clean_punct(df, "?!.,()/+:;<=>[]{|}%&–~\"")
        df["char_word"] = df.comment_text.map(self.char_to_word)
        df.comment_text[df.char_word.notnull()] = df[df.char_word.notnull()].apply(
            lambda x: self.word_regex(x, embeddings_index), axis=1)
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
        word_dict = json.loads(open("../input/basedata20190603/word_dict.json").read())
        replace_dict = dict(replace_dict, **word_dict)
        df["comment_text"] = df.comment_text.map(lambda x: self.word_replace(x, replace_dict))
        df = self.clean_punct(df, "'*$^@#")
        longword_dict = json.loads(open("../input/basedata20190603/longword_dict.json").read())
        spell_dict = json.loads(open("../input/basedata20190603/spell_dict.json").read())
        replace_dict = dict(replace_dict, **longword_dict, **spell_dict)
        df["comment_text"] = df.comment_text.map(lambda x: self.word_replace(x, replace_dict))
        if debug:
            pass
        else:
            word_merge = json.loads(open("../input/basedata20190603/word_merge.json").read())

            def bool_series_create(x):

                if sum([1 if key in x else 0 for key in word_merge.keys()]) > 0:
                    return True
                else:
                    return False

            bool_series = df.comment_text.map(bool_series_create)
            df.comment_text[bool_series] = df.comment_text[bool_series].map(lambda x: self.multiple_replace(x, word_merge))
        self.train = pd.read_csv("../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv")
        self.test = pd.read_csv("../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv")
        if debug:
            train = self.train.iloc[:10000]
            test = self.test.iloc[:10000]
        self.train["comment_text"] = df[df.target.notnull()].comment_text.values
        self.test["comment_text"] = df[df.target.isnull()].comment_text.values
        del embeddings_index, df
        gc.collect()
