import torch
import torch.nn.functional as F
from transformers import BertTokenizer
from tqdm import tqdm
import pandas as pd
import numpy as np

from .base_tester import Tester
from textclf.utils.raw_data import create_tokenizer
from textclf.utils.create import create_instance
from textclf.config import DLTesterConfig, DataLoaderConfig
from textclf.data.dictionary import Dictionary
from textclf.data.loader import build_loader


class DLTester(Tester):
    """负责Deep Learning model的测试"""

    def __init__(self, config: DLTesterConfig):
        super().__init__(config)
        self.tokenizer = create_tokenizer(self.config.tokenizer)
        self.use_cuda = self.config.use_cuda and torch.cuda.is_available()

        print(f"use_cuda: {self.use_cuda}..")
        if not self.use_cuda:
            print(f"Load checkpoint from {self.config.model_path} to cpu..")
            checkpoint = torch.load(self.config.model_path, map_location=torch.device('cpu'))
        else:
            print(f"Load checkpoint from {self.config.model_path} to GPU..")
            checkpoint = torch.load(self.config.model_path, map_location=torch.device('cuda:0'))

        self.model_conf, self.dictionary, self.label2id = \
            checkpoint["info_for_test"]
        print("Model conf:", self.model_conf)
        self.model_conf.embedding_layer.pretrained_path = None
        self.model_conf.embedding_layer.method = "random"

        self.model = create_instance(self.model_conf)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if self.use_cuda:
            self.model = self.model.cuda()
        self.classes = sorted(self.label2id, key=self.label2id.get)

    def _preprocess(self, text):
        text_tokenized = self.tokenizer(text)
        if isinstance(self.dictionary, Dictionary):
            # text_processed = self.dictionary.tokens_to_tensor(
            #     text_tokenized, max_len=min(self.config.max_len, len(text_tokenized))
            # )
            text_processed = self.dictionary.tokens_to_tensor(
                text_tokenized, max_len=self.config.max_len
            )
            text_len = (text_processed != self.dictionary.pad()).sum()
            # text_len = torch.LongTensor(self.config.max_len)
        elif isinstance(self.dictionary, BertTokenizer):
            text_processed = torch.LongTensor(
                self.dictionary.encode(text_tokenized, add_special_tokens=True)[:-1])
            max_len = self.config.max_len
            pad_id = self.dictionary.pad_token_id
            if len(text_processed) >= max_len:
                text_processed = text_processed[:max_len]
            else:
                text_processed = torch.cat([
                    text_processed,
                    torch.ones(max_len - len(text_processed)).long() * pad_id
                ])
            text_len = (text_processed != pad_id).sum()

        if self.use_cuda:
            text_processed = text_processed.cuda()
            text_len = text_len.cuda()

        return text_processed.unsqueeze(0), text_len.unsqueeze(0)

    def predict_batch(self, texts):
        """
        predict top1 label and score
        """
        text_ids, text_lens = [], []
        for text in texts:
            text_id, text_len = self._preprocess(text)
            text_ids.append(text_id)
            text_lens.append(text_len)
        text_ids = torch.concat(text_ids)
        text_lens = torch.concat(text_lens)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(text_ids, text_lens)
        probs = torch.softmax(logits, dim=1)
        scores, label_ids = torch.max(probs, dim=1)
        labels = [self.classes[idx] for idx in label_ids.tolist()]
        return labels, scores.tolist()

    def predict_label(self, text):
        text_processed, text_len = self._preprocess(text)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(text_processed, text_len)
        label_id = torch.argmax(logits)
        return self.classes[label_id]

    def predict_prob(self, text):
        text_processed, text_len = self._preprocess(text)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(text_processed, text_len)
            if logits.dim() == 2:
                logits = logits[0]
        return torch.softmax(logits, dim=0).tolist()

    def get_all_labels(self):
        return self.classes

    def test_file(self):
        """对给定的文件进行预测"""

        inp = open(self.config.input_file)

        has_target = self.config.has_target
        predict_prob = self.config.predict_prob
        write_badcase = (self.config.badcase_file is not None) and has_target
        batch_size = self.config.batch_size

        loader_config = DataLoaderConfig()
        loader_config.batch_size = batch_size
        loader_config.max_len = self.config.max_len
        loader_config.drop_last = False
        loader_config.shuffle = False

        pairs = []
        dummy_target = list(self.label2id.keys())[0]
        for line in inp:
            if has_target:
                text, target = line.strip().split('\t')
            else:
                text = line.strip()
                target = dummy_target
            pairs.append((" ".join(list(text)), target))

        columns = ["text", "predict", "target", "loss"] + self.get_all_labels()
        size = len(columns)
        result_df = pd.DataFrame(np.zeros((len(pairs), size)), columns=columns)
        result_df["text"] = [p[0] for p in pairs]
        result_df["target"] = [p[1] for p in pairs]

        loader = build_loader(pairs, self.dictionary, self.label2id, loader_config)
        predicts = []
        probs = []
        self.model.eval()
        losses = []
        with torch.no_grad():
            for batch_id, batch in tqdm(enumerate(loader, 1)):
                text_tensor, text_lens, labels = batch
                # n_padding = self.config.max_len - text_tensor.size(1)
                # if n_padding > 0:
                #     padding = torch.zeros(
                #         [batch_size, self.config.max_len], dtype=text_tensor.dtype)
                #     text_tensor = torch.concat([text_tensor, padding], dim=1)

                if self.use_cuda:
                    text_tensor = text_tensor.cuda()
                    text_lens = text_lens.cuda()
                    labels = labels.cuda()
                logits = self.model(text_tensor, text_lens)

                if has_target:
                    loss = F.cross_entropy(logits, labels, reduction="none")
                    losses.extend(loss.cpu().tolist())

                label_ids = torch.argmax(logits, dim=1)
                predicts.extend([self.classes[id_] for id_ in label_ids.tolist()])
                batch_probs = torch.softmax(logits, dim=1)
                probs.append(batch_probs)

        probs = torch.cat(probs)
        result_df["predict"] = predicts
        if has_target:
            result_df["loss"] = losses
        result_df.loc[:, self.get_all_labels()] = probs.cpu().numpy()

        if predict_prob:
            out_columns = columns
        else:
            out_columns = ["text", "target", "predict", "loss"]

        if not has_target:
            out_columns.remove("target")
            out_columns.remove("loss")

        acc = (result_df['predict'] == result_df['target']).sum() / len(result_df)
        avg_loss = result_df['loss'].sum() / len(result_df)
        print(f"ACC: {acc:.4f}")
        print(f"Avg Loss: {avg_loss:.4f}")
        if self.config.out_file:
            result_df.to_csv(self.config.out_file, sep="\t",
                             index=None, columns=out_columns)
        if write_badcase:
            badcase = result_df[result_df["target"] != result_df["predict"]]
            badcase.to_csv(self.config.badcase_file, sep="\t", index=None,
                           columns=out_columns)
        return acc, avg_loss
