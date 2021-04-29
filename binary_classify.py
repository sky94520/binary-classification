import time
import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.metrics import classification_report
from concurrent.futures import ThreadPoolExecutor
from torch.utils.data import TensorDataset, DataLoader
from pytorch_pretrained_bert import BertTokenizer, BertModel
from pytorch_pretrained_bert.optimization import BertAdam


class ClassifyModel(nn.Module):
    def __init__(self, pretrained_model_name_or_path, num_labels, is_lock=False):
        super(ClassifyModel, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name_or_path)
        config = self.bert.config
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(768, num_labels)
        if is_lock:
            # 加载并冻结bert模型参数
            for name, param in self.bert.named_parameters():
                if name.startswith('pooler'):
                    continue
                else:
                    param.requires_grad_(False)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        _, pooled = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        # 测试1，直接
        return logits


class DataProcessForSingleSentence(object):
    def __init__(self, bert_tokenizer, max_workers=10):
        """
        :param bert_tokenizer: 分词器
        :param max_workers:  包含列名comment和sentiment的data frame
        """
        self.bert_tokenizer = bert_tokenizer
        self.pool = ThreadPoolExecutor(max_workers=max_workers)

    def get_input(self, dataset, max_seq_len=30):
        sentences = dataset.iloc[:, 1].tolist()
        labels = dataset.iloc[:, 2].tolist()
        # 切词
        token_seq = list(self.pool.map(self.bert_tokenizer.tokenize, sentences))
        # 获取定长序列及其mask
        result = list(self.pool.map(self.trunate_and_pad, token_seq,
                                    [max_seq_len] * len(token_seq)))
        seqs = [i[0] for i in result]
        seq_masks = [i[1] for i in result]
        seq_segments = [i[2] for i in result]

        t_seqs = torch.tensor(seqs, dtype=torch.long)
        t_seq_masks = torch.tensor(seq_masks, dtype=torch.long)
        t_seq_segments = torch.tensor(seq_segments, dtype=torch.long)
        t_labels = torch.tensor(labels, dtype=torch.long)

        return TensorDataset(t_seqs, t_seq_masks, t_seq_segments, t_labels)

    def trunate_and_pad(self, seq, max_seq_len):
        # 对超长序列进行截断
        if len(seq) > (max_seq_len - 2):
            seq = seq[0: (max_seq_len - 2)]
            # 添加特殊字符
        seq = ['[CLS]'] + seq + ['[SEP]']
        # id化
        seq = self.bert_tokenizer.convert_tokens_to_ids(seq)
        # 根据max_seq_len与seq的长度产生填充序列
        padding = [0] * (max_seq_len - len(seq))
        # 创建seq_mask
        seq_mask = [1] * len(seq) + padding
        # 创建seq_segment
        seq_segment = [0] * len(seq) + padding
        # 对seq拼接填充序列
        seq += padding
        assert len(seq) == max_seq_len
        assert len(seq_mask) == max_seq_len
        assert len(seq_segment) == max_seq_len
        return seq, seq_mask, seq_segment


def load_data(filepath, pretrained_model_name_or_path, max_seq_len, batch_size):
    """
    加载excel文件，有train和test 的sheet
    :param filepath: 文件路径
    :param pretrained_model_name_or_path: 使用什么样的bert模型
    :param max_seq_len: bert最大尺寸，不能超过512
    :param batch_size: 小批量训练的数据
    :return: 返回训练和测试数据迭代器 DataLoader形式
    """
    io = pd.io.excel.ExcelFile(filepath)
    raw_train_data = pd.read_excel(io, sheet_name='train')
    raw_test_data = pd.read_excel(io, sheet_name='test')
    io.close()
    # 分词工具
    bert_tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path, do_lower_case=True)
    processor = DataProcessForSingleSentence(bert_tokenizer=bert_tokenizer)
    # 产生输入句 数据
    train_data = processor.get_input(raw_train_data, max_seq_len)
    test_data = processor.get_input(raw_test_data, max_seq_len)

    train_iter = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_iter = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)
    return train_iter, test_iter


def evaluate_accuracy(data_iter, net, device):
    # 记录预测标签和真实标签
    prediction_labels, true_labels = [], []
    with torch.no_grad():
        for batch_data in data_iter:
            batch_data = tuple(t.to(device) for t in batch_data)
            # 获取给定的输出和模型给的输出
            labels = batch_data[-1]
            output = net(*batch_data[:-1])
            predictions = output.softmax(dim=1).argmax(dim=1)
            prediction_labels.append(predictions.detach().cpu().numpy())
            true_labels.append(labels.detach().cpu().numpy())

    return classification_report(np.concatenate(true_labels), np.concatenate(prediction_labels))


if __name__ == '__main__':
    batch_size, max_seq_len = 32, 200
    train_iter, test_iter = load_data('dianping_train_test.xls', 'bert-base-chinese', max_seq_len, batch_size)
    # 加载模型
    # model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=2)
    model = ClassifyModel('bert-base-chinese', num_labels=2, is_lock=True)
    print(model)

    optimizer = BertAdam(model.parameters(), lr=5e-05)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    loss_func = nn.CrossEntropyLoss()

    for epoch in range(4):
        start = time.time()
        model.train()
        # loss和精确度
        train_loss_sum, train_acc_sum, n = 0.0, 0.0, 0
        for step, batch_data in enumerate(train_iter):
            batch_data = tuple(t.to(device) for t in batch_data)
            batch_seqs, batch_seq_masks, batch_seq_segments, batch_labels = batch_data

            logits = model(batch_seqs, batch_seq_masks, batch_seq_segments)
            logits = logits.softmax(dim=1)
            loss = loss_func(logits, batch_labels)
            loss.backward()
            train_loss_sum += loss.item()
            train_acc_sum += (logits.argmax(dim=1) == batch_labels).sum().item()
            n += batch_labels.shape[0]
            optimizer.step()
            optimizer.zero_grad()
        # 每一代都判断
        model.eval()

        result = evaluate_accuracy(test_iter, model, device)
        print('epoch %d, loss %.4f, train acc %.3f, time: %.3f' %
              (epoch + 1, train_loss_sum / n, train_acc_sum / n, (time.time() - start)))
        print(result)

    torch.save(model, 'fine_tuned_chinese_bert.bin')
