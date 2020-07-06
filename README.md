# top67-solution-for-Jigsaw-Unintended-Bias-in-Toxicity-Classification
kaggle-Jigsaw银牌记录
## 数据清洗：
##### 统一用户输入的多种不规范字符（以免bert从语料库中查找到不同的id）
```python
#目标字典
E_pun = 'BEFIKLMOQSTWZBCFGJKLMPVWXZCILOABCDFGHJKLMNOPQRSUVWXYABCDEHIJKLMOPRTWYBJKMVXZ012345678901234567FWY,;.!?:$ []()<>""\'\'AAAAAAAABBCCDDDDEEEEEEEEFFGGGHHHHIIIIIIJKLLLMNNNNNOOOOOOOPPRRRSSSSTTTTTTUUUUVWWYYYZPABBDEGHIJKLMNOPRTUWabdegkmnNoptuwvhjrwyxylsx\'\'    ' + 'ABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZ'
#整理的不规则字符
C_pun = '𝗕𝗘𝗙𝗜𝗞𝗟𝗠𝗢𝗤𝗦𝗧𝗪𝗭𝗯𝗰𝗳𝗴𝗷𝗸𝗹𝗺𝗽𝘃𝘄𝘅𝘇𝘊𝘐𝘓𝘖𝘢𝘣𝘤𝘥𝘧𝘨𝘩𝘫𝘬𝘭𝘮𝘯𝘰𝘱𝘲𝘳𝘴𝘶𝘷𝘸𝘹𝘺𝘼𝘽𝘾𝘿𝙀𝙃𝙄𝙅𝙆𝙇𝙈𝙊𝙋𝙍𝙏𝙒𝙔𝙗𝙟𝙠𝙢𝙫𝙭𝙯𝟎𝟏𝟐𝟑𝟒𝟓𝟔𝟕𝟖𝟗𝟬𝟭𝟮𝟯𝟰𝟱𝟲𝟳🇫🇼🇾，；。！？：￥—【】（）《》“”‘’𝙖𝗮äаāàáᴀᴃʙᴄ𝙘ᴅ𝙙𝗱ᴆ𝙚𝖊𝘦𝗲éèêᴇғ𝙛ʛɢ𝙜ʜн𝗵𝙝í𝗶𝘪𝙞ïɪᴊᴋ𝙡ʟᴌᴍᴎ𝙣𝗻ñ𝖓ᴏо𝖔𝗼𝙤öóᴘ𝙥ʀ𝙧𝗿𝘀ѕś𝙨ᴛт𝖙𝘵𝘁𝙩ū𝙪𝘂ᴜᴠᴡ𝙬ʏ𝙮𝘆ᴢᴩᴬᴮᴯᴰᴱᴳᴴᴵᴶᴷᴸᴹᴺɴᴼᴾᴿᵀᵁᵂᵃᵇᵈᵉᵍᵏᵐᵑᵒᵖᵗᵘᵚᵛʰʲʳʷʸ˟ˠˡˢˣ´`…-_•' + '𝐀𝐁𝐂𝐃𝐄𝐅𝐆𝐇𝐈𝐉𝐊𝐋𝐌𝐍𝐎𝐏𝐐𝐑𝐒𝐓𝐔𝐕𝐖𝐗𝐘𝐙𝐚𝐛𝐜𝐝𝐞𝐟𝐠𝐡𝐢𝐣𝐤𝐥𝐦𝐧𝐨𝐩𝐪𝐫𝐬𝐭𝐮𝐯𝐰𝐱𝐲𝐳𝐴𝐵𝐶𝐷𝐸𝐹𝐺𝐻𝐼𝐽𝐾𝐿𝑀𝑁𝑂𝑃𝑄𝑅𝑆𝑇𝑈𝑉𝑊𝑋𝑌𝑍𝑎𝑏𝑐𝑑𝑒𝑓𝑔𝑖𝑗𝑘𝑙𝑚𝑛𝑜𝑝𝑞𝑟𝑠𝑡𝑢𝑣𝑤𝑥𝑦𝑧𝑨𝑩𝑪𝑫𝑬𝑭𝑮𝑯𝑰𝑱𝑲𝑳𝑴𝑵𝑶𝑷𝑸𝑹𝑺𝑻𝑼𝑽𝑾𝑿𝒀𝒁𝒂𝒃𝒄𝒅𝒆𝒇𝒈𝒉𝒊𝒋𝒌𝒍𝒎𝒏𝒐𝒑𝒒𝒓𝒔𝒕𝒖𝒗𝒘𝒙𝒚𝒛'
```
##### 替换读入格式造成的乱码为正常数据
```python
df["comment_text"] = df.comment_text.str.replace("·", "").str.replace("\xad", "")
df["comment_text"] = df.comment_text.replace("(U|u)\.(S|s)\.(A|a)", "USA")
df["comment_text"] = df.comment_text.map(lambda x: re.sub("((ht|f)tps?):\/\/[^\s]*", "http ", x))
df["comment_text"] = df.comment_text.map(lambda x: re.sub("[\u0800-\u4e00\uAC00-\uD7A3\u4E00-\u9FA5]", "", x))
```
##### 清理标点符号和被空格错误分割的词语
```python
#标点符号前后应有空格
df = self.clean_punct(df, "?!.,()/+:;<=>[]{|}%&–~\"")
#例如：c o o l 应被视为cool
df["char_word"] = df.comment_text.map(self.char_to_word)
df.comment_text[df.char_word.notnull()] = df[df.char_word.notnull()].apply(
    lambda x: self.word_regex(x, embeddings_index), axis=1)
df = df.drop(["char_word"], axis=1)
```

