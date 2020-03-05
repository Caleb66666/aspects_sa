import os
import pandas as pd
import numpy as np
from data_loader.base_loader import BaseLoader
from utils.path_util import nps_deserialize, nps_serialize
from custom_modules.albert import AlbertTokenizer


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

        albert_tokenizer = AlbertTokenizer.from_pretrained(self.config.transfer_path)
        train_df, valid_df, albert_tokenizer = self.tokenize_and_stop_symbols(
            train_df=train_df,
            valid_df=valid_df,
            debug=self.config.debug,
            tokenizer=albert_tokenizer,
            premise=self.config.premise,
            tokens_col=self.tokens_col,
        )

        train_df, valid_df = self.index_pad_truncate(
            train_df=train_df,
            valid_df=valid_df,
            debug=self.config.debug,
            max_seq=self.config.max_seq,
            tokenizer=albert_tokenizer,
            tokens_col=self.tokens_col,
            ids_col=self.ids_col,
            len_col=self.len_col,
            mask_col=self.mask_col,
            inf_mask_col=self.inf_mask_col,
            truncate_method=self.config.truncate_method
        )

        train_df, valid_df = self.closeout_process(
            train_df=train_df,
            valid_df=valid_df,
            debug=self.config.debug,
            dropped_columns=["id", self.config.premise, self.tokens_col]
        )

        nps_serialize(self.config.processed_np, train_df.to_numpy(), valid_df.to_numpy(),
                      np.array(train_df.columns.tolist()))

    def workflow(self):
        if not os.path.exists(self.config.processed_np):
            self._workflow()

        train_np, valid_np, columns_np = nps_deserialize(self.config.processed_np)
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
        self.config.feature_cols = [self.ids_col, self.len_col, self.mask_col, self.inf_mask_col]
        self.config.num_labels = len(columns) - len(self.config.feature_cols)

        return train_batches, valid_batches
