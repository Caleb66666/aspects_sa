# @File: new_word_char_loader
# @Author : Caleb
# @Email: VanderLancer@gmail.com
# @time: 2020/2/29 15:59:28

import os
from data_loader.base_loader import BaseLoader
from utils.path_util import serialize, deserialize, nps_serialize, nps_deserialize
from utils.time_util import timer
import pandas as pd


class TrainLoader(BaseLoader):
    word_tokens_col = "word_tokens"
    word_ids_col = "word_ids"
    char_tokens_col = "char_tokens"
    char_ids_col = "char_ids"
    len_col = "seq_len"
    mask_col = "seq_mask"
    inf_mask_col = "inf_mask"

    def __init__(self, config):
        super(TrainLoader, self).__init__(config.nb_workers)
        self.config = config

        self.train_batches, self.valid_batches = self.workflow()

    @timer
    def _workflow(self):
        train_df, valid_df = self.read_raw(
            files=(self.config.train_file, self.config.valid_file),
            header=self.config.header,
            sep=self.config.sep,
            debug=self.config.debug,
            shuffle=self.config.shuffle,
            seed=self.config.seed,
            encoding=self.config.encoding
        )
        train_df, valid_df = self.pretreatment_text(
            train_df=train_df,
            valid_df=valid_df,
            premise=self.config.premise,
            debug=self.config.debug,
            if_lower=self.config.if_lower
        )

        train_df, valid_df, word_tokenizer = self.tokenize_and_stop_symbols(
            train_df=train_df,
            valid_df=valid_df,
            debug=self.config.debug,
            max_vocab=self.config.word_max_vocab,
            pad_token=self.config.word_pad_token,
            unk_token=self.config.word_unk_token,
            tokenize_method="word",
            stop_symbols_file=self.config.stop_dict,
            premise=self.config.premise,
            tokens_col=self.word_tokens_col,
            user_dict=self.config.user_dict,
            min_count=self.config.word_min_count
        )

        train_df, valid_df, char_tokenizer = self.tokenize_and_stop_symbols(
            train_df=train_df,
            valid_df=valid_df,
            debug=self.config.debug,
            max_vocab=self.config.char_max_vocab,
            pad_token=self.config.char_pad_token,
            unk_token=self.config.char_unk_token,
            tokenize_method="char",
            stop_symbols_file=self.config.stop_dict,
            premise=self.config.premise,
            tokens_col=self.char_tokens_col,
            min_count=self.config.char_min_count,
        )

        train_df, valid_df = self.index_pad_truncate_word_char(
            train_df=train_df,
            valid_df=valid_df,
            debug=self.config.debug,
            max_seq=self.config.max_seq,
            word_tokenizer=word_tokenizer,
            char_tokenizer=char_tokenizer,
            word_col=self.word_tokens_col,
            word_ids_col=self.word_ids_col,
            char_ids_col=self.char_ids_col,
            len_col=self.len_col,
            mask_col=self.mask_col,
            inf_mask_col=self.inf_mask_col,
            char_limit=self.config.char_limit,
            truncate_method=self.config.truncate_method
        )

        word_embed = self.embed_tokens(
            w2v_path=self.config.word_w2v,
            train_df=train_df,
            valid_df=valid_df,
            tokens_col=self.word_tokens_col,
            seed=self.config.seed,
            embed_dim=self.config.word_embed_dim,
            max_vocab=self.config.word_max_vocab,
            pad_token=self.config.word_pad_token,
            unk_token=self.config.word_unk_token,
            window=self.config.word_window,
            min_count=self.config.word_min_count,
            workers=self.config.nb_workers,
            iterations=self.config.word_iterations,
            tokenizer=word_tokenizer,
        )

        char_embed = self.embed_tokens(
            w2v_path=self.config.char_w2v,
            train_df=train_df,
            valid_df=valid_df,
            tokens_col=self.char_tokens_col,
            seed=self.config.seed,
            embed_dim=self.config.char_embed_dim,
            max_vocab=self.config.char_max_vocab,
            pad_token=self.config.char_pad_token,
            unk_token=self.config.char_unk_token,
            window=self.config.char_window,
            min_count=self.config.char_min_count,
            workers=self.config.nb_workers,
            iterations=self.config.char_iterations,
            tokenizer=char_tokenizer
        )

        train_df, valid_df = self.closeout_process(
            train_df=train_df,
            valid_df=valid_df,
            debug=self.config.debug,
            dropped_columns=["id", self.config.premise, self.word_tokens_col, self.char_tokens_col]
        )
        serialize(self.config.dl_path, (word_tokenizer, char_tokenizer, train_df.columns.tolist()))
        nps_serialize(self.config.processed_np, train_df.to_numpy(), valid_df.to_numpy(), word_embed, char_embed)

    def workflow(self):
        if not os.path.exists(self.config.processed_np):
            self._workflow()
        word_tokenizer, char_tokenizer, columns = deserialize(self.config.dl_path)
        train_np, valid_np, word_embed, char_embed = nps_deserialize(self.config.processed_np)

        train_df = pd.DataFrame(train_np, columns=columns)
        valid_df = pd.DataFrame(valid_np, columns=columns)
        del train_np, valid_np

        long_field, float_field, label_field, fields = self.prepare_fields_word_char(
            columns=columns,
            word_ids_col=self.word_ids_col,
            char_ids_col=self.char_ids_col,
            len_col=self.len_col,
            mask_col=self.mask_col,
            inf_mask_col=self.inf_mask_col
        )
        train_ds, valid_ds = self.df2ds(train_df, fields, columns), self.df2ds(valid_df, fields, columns)
        label_field.build_vocab(train_ds)
        train_batches, valid_batches = self.new_batch_data(
            train_ds=train_ds,
            valid_ds=valid_ds,
            columns=columns,
            batch_size=self.config.batch_size,
            device=self.config.device,
            sort_within_batch=self.config.sort_within_batch,
            len_column=self.len_col,
        )
        self.config.classes = list(label_field.vocab.stoi.values())
        self.config.num_classes = len(self.config.classes)
        self.config.word_embed = word_embed
        self.config.char_embed = char_embed
        self.config.feature_cols = [self.word_ids_col, self.char_ids_col, self.len_col, self.mask_col,
                                    self.inf_mask_col]
        self.config.num_labels = len(columns) - len(self.config.feature_cols)
        self.config.word_vocab_size = len(word_tokenizer.word2index)
        self.config.char_vocab_size = len(char_tokenizer.word2index)
        return train_batches, valid_batches


if __name__ == '__main__':
    import sys

    sys.path.append("..")
    from models.word_char_pool import Config

    config_ = Config(debug=True)
    loader = TrainLoader(config_)
