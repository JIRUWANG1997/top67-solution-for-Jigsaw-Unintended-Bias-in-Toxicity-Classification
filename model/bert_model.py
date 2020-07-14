import numpy as np
from pytorch_pretrained_bert import BertTokenizer, BertForSequenceClassification, BertAdam
from pytorch_pretrained_bert import BertConfig
import torch
def convert_lines(self, example, max_seq_length, tokenizer):
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

class bert_model():
    def bert_predict(test_df, BERT_MODEL_PATH, bert_path_list):
        tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH, cache_dir=None, do_lower_case=True)
        X_test = convert_lines(test_df["comment_text"].fillna("DUMMY_VALUE"), 220, tokenizer)
        test_pred_list = []
        for bert_path in bert_path_list:
            bert_config = BertConfig(bert_path + 'bert_config.json')
            device = torch.device('cuda')
            model = BertForSequenceClassification(bert_config, num_labels=8)
            model.load_state_dict(torch.load(bert_path + "bert_pytorch.bin"))
            model.to(device)
            for param in model.parameters():
                param.requires_grad = False
            model.eval()

            test_preds = np.zeros((len(X_test)))
            test = torch.utils.data.TensorDataset(torch.tensor(X_test, dtype=torch.long))
            test_loader = torch.utils.data.DataLoader(test, batch_size=512, shuffle=False)
            for i, (x_batch,) in enumerate(test_loader):
                pred = model(x_batch.to(device), attention_mask=(x_batch > 0).to(device), labels=None)
                test_preds[i * 512:(i + 1) * 512] = pred[:, 0].detach().cpu().squeeze().numpy()
            test_pred = torch.sigmoid(torch.tensor(test_preds)).numpy().ravel()
            test_pred_list.append(test_pred)
        result = np.average(test_pred_list, weights=[1, 2], axis=0)
        return test_pred

