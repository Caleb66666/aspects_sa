# @File: word_loader
# @Author : Caleb
# @Email: VanderLancer@gmail.com
# @time: 2020/3/2 18:00:41

import os
import pandas as pd
import numpy as np
from data_loader.base_loader import BaseLoader
from utils.path_util import nps_deserialize, nps_serialize
from utils.vocab_util import Tokenizer


class TrainLoader(BaseLoader):
    tokens_col = "tokens"
    ids_col = "seq_ids"
    len_col = "seq_len"
    mask_col = "seq_mask"
    inf_mask_col = "inf_mask"

    def __init__(self, config):
        super(TrainLoader, self).__init__(config.nb_workers)
        self.config = config
        self.train_batches, self.valid_batches = self.workflow()

    def _workflow(self):
        train_df, valid_df = self.read_raw_and_pretreatment(
            debug=self.config.debug,
            files=(self.config.train_file, self.config.valid_file),
            header=self.config.header,
            sep=self.config.sep,
            shuffle=self.config.shuffle,
            seed=self.config.seed,
            encoding=self.config.encoding,
            premise=self.config.premise,
            if_lower=self.config.if_lower,
            processed_train=self.config.processed_train,
            processed_valid=self.config.processed_valid,
        )

        train_df, valid_df, word_tokenizer = self.tokenize_and_stop_symbols(
            train_df=train_df,
            valid_df=valid_df,
            debug=self.config.debug,
            tokenizer_path=self.config.word_tokenizer_path,
            max_vocab_size=self.config.word_max_vocab,
            pad_token=self.config.word_pad_token,
            unk_token=self.config.word_unk_token,
            split_type="word",
            stop_symbols_file=None,
            premise=self.config.premise,
            tokens_col=self.tokens_col,
            user_dict=self.config.user_dict,
            min_count=self.config.word_min_count
        )

        train_df, valid_df = self.index_pad_truncate(
            train_df=train_df,
            valid_df=valid_df,
            debug=self.config.debug,
            max_seq=self.config.max_seq,
            tokenizer=word_tokenizer,
            tokens_col=self.tokens_col,
            ids_col=self.ids_col,
            len_col=self.len_col,
            mask_col=self.mask_col,
            inf_mask_col=self.inf_mask_col,
            truncate_method=self.config.truncate_method,
        )

        word_embed_mat = self.gen_embed_matrix(
            w2v_path=self.config.word_w2v,
            vocab=word_tokenizer.vocab,
            train_df=train_df,
            valid_df=valid_df,
            tokens_col=self.tokens_col,
            seed=self.config.seed,
            embed_dim=self.config.word_embed_dim,
            max_vocab_size=self.config.word_max_vocab,
            unk_token=self.config.word_unk_token,
            pad_token=self.config.word_pad_token,
            window=self.config.word_window,
            min_count=self.config.word_min_count,
            workers=self.config.nb_workers,
            iterations=self.config.word_iterations,
        )

        train_df, valid_df = self.closeout_process(
            train_df=train_df,
            valid_df=valid_df,
            debug=self.config.debug,
            dropped_columns=["id", self.config.premise, self.tokens_col]
        )

        nps_serialize(self.config.processed_np, train_df.to_numpy(), valid_df.to_numpy(), word_embed_mat,
                      np.array(train_df.columns.tolist()))

    def workflow(self):
        if not os.path.exists(self.config.processed_np):
            self._workflow()

        word_tokenizer = Tokenizer.load(self.config.word_tokenizer_path)
        train_np, valid_np, word_embed_matrix, columns_np = nps_deserialize(self.config.processed_np)
        columns = columns_np.tolist()
        train_df = pd.DataFrame(train_np, columns=columns)
        valid_df = pd.DataFrame(valid_np, columns=columns)
        del train_np, valid_np, columns_np

        long_field, float_field, label_field, fields = self.prepare_fields(
            columns=columns,
            ids_column=self.ids_col,
            len_column=self.len_col,
            mask_column=self.mask_col,
            inf_mask_column=self.inf_mask_col
        )

        train_ds, valid_ds = self.df2ds(train_df, fields, columns), self.df2ds(valid_df, fields, columns)
        label_field.build_vocab(train_ds)

        train_batches, valid_batches = self.batch_data(
            train_ds=train_ds,
            valid_ds=valid_ds,
            columns=columns,
            batch_size=self.config.batch_size,
            device=self.config.device,
            sort_within_batch=self.config.sort_within_batch,
            len_column=self.len_col
        )
        self.config.classes = list(label_field.vocab.stoi.values())
        self.config.num_classes = len(self.config.classes)
        self.config.word_embed_matrix = word_embed_matrix
        self.config.feature_cols = [self.ids_col, self.len_col, self.mask_col, self.inf_mask_col]
        self.config.num_labels = len(columns) - len(self.config.feature_cols)
        self.config.word_vocab_size = word_tokenizer.vocab_size
        return train_batches, valid_batches


if __name__ == '__main__':
    import sys

    sys.path.append("..")
    from models.word_char_pool import Config

    config_ = Config(debug=True)
    loader = TrainLoader(config_)
