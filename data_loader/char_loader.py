# @File: char_loader
# @Author : Caleb
# @Email: VanderLancer@gmail.com
# @time: 2020/2/18 16:53:55

import os
from data_loader.base_loader import BaseLoader
from utils.path_util import serialize, deserialize


class TrainLoader(BaseLoader):
    tokens_column = "char_tokens"
    ids_column = "seq_ids"
    len_column = "seq_len"
    mask_column = "seq_mask"
    inf_mask_column = "inf_mask"

    def __init__(self, config):
        super(TrainLoader, self).__init__(config.nb_workers)
        self.config = config

        if os.path.exists(self.config.dl_path):
            train_df, valid_df, fields, label_field, columns, embed_matrix = deserialize(self.config.dl_path)
        else:
            train_df, valid_df, fields, label_field, columns, embed_matrix = self.workflow()

        self.train_batches, self.valid_batches = self.batch_data(
            train_df=train_df,
            valid_df=valid_df,
            fields=fields,
            columns=columns,
            batch_size=self.config.batch_size,
            device=self.config.device,
            sort_within_batch=self.config.sort_within_batch,
            len_column=self.len_column,
            build_vocab_field=label_field,
        )

        config.classes = list(label_field.vocab.stoi.values())
        config.num_classes = len(config.classes)
        config.embed_matrix = embed_matrix
        config.num_labels = len(columns) - len(
            [self.ids_column, self.len_column, self.mask_column, self.inf_mask_column])

    def workflow(self):
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

        train_df, valid_df, tokenizer = self.tokenize_and_stop_symbols(
            train_df=train_df,
            valid_df=valid_df,
            debug=self.config.debug,
            max_vocab=self.config.max_vocab,
            pad_token=self.config.pad_token,
            unk_token=self.config.unk_token,
            tokenize_method=self.config.tokenize_method,
            stop_symbols_file=self.config.stop_dict,
            premise=self.config.premise,
            tokens_col=self.tokens_column,
        )

        train_df, valid_df = self.index_pad_truncate(
            train_df=train_df,
            valid_df=valid_df,
            debug=self.config.debug,
            max_seq=self.config.max_seq,
            tokenizer=tokenizer,
            tokens_col=self.tokens_column,
            ids_col=self.ids_column,
            len_col=self.len_column,
            mask_col=self.mask_column,
            inf_mask_col=self.inf_mask_column,
            truncate_method=self.config.truncate_method
        )

        embed_matrix = self.embed_tokens(
            w2v_path=self.config.w2v_path,
            train_df=train_df,
            valid_df=valid_df,
            tokens_col=self.tokens_column,
            seed=self.config.seed,
            embed_dim=self.config.embed_dim,
            max_vocab=self.config.max_vocab,
            pad_token=self.config.pad_token,
            unk_token=self.config.unk_token,
            window=self.config.window,
            min_count=self.config.min_count,
            workers=self.config.nb_workers,
            iterations=self.config.iterations,
            tokenizer=tokenizer,
            word2index_attr="word2index"
        )

        train_df, valid_df = self.closeout_process(
            train_df=train_df,
            valid_df=valid_df,
            debug=self.config.debug,
            dropped_columns=["id", self.config.premise, self.tokens_column]
        )

        columns = train_df.columns.tolist()
        long_field, float_field, label_field, fields = self.prepare_fields(
            columns=columns,
            ids_column=self.ids_column,
            len_column=self.len_column,
            mask_column=self.mask_column,
            inf_mask_column=self.inf_mask_column
        )

        target_obj = (train_df, valid_df, fields, label_field, columns, embed_matrix)
        serialize(self.config.dl_path, target_obj)
        return target_obj


if __name__ == '__main__':
    from models.rcnn_model import Config

    config_ = Config(debug=True)
    loader = TrainLoader(config_)
    print(config_.classes)
    print(config_.num_labels)
