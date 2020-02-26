# @File: transfer_loader
# @Author : Caleb
# @Email: VanderLancer@gmail.com
# @time: 2020/2/22 12:09:40

import os
from data_loader.base_loader import BaseLoader
from utils.path_util import deserialize, serialize
from custom_modules.albert import AlbertTokenizer


class TrainLoader(BaseLoader):
    tokens_column = "tokens"
    ids_column = "seq_ids"
    len_column = "seq_len"
    mask_column = "seq_mask"
    inf_mask_column = "inf_mask"

    def __init__(self, config):
        super(TrainLoader, self).__init__(config.nb_workers)
        self.config = config

        if os.path.exists(self.config.dl_path):
            train_df, valid_df, fields, label_field, columns = deserialize(self.config.dl_path)
        else:
            train_df, valid_df, fields, label_field, columns = self.workflow()

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

        train_df, valid_df = self.pretreatment_text(train_df, valid_df, self.config.premise, self.config.debug)

        tokenizer = AlbertTokenizer.from_pretrained(self.config.transfer_path)
        train_df, valid_df, if_custom_tokenizer, tokenizer = self.tokenize_and_index(
            train_df=train_df,
            valid_df=valid_df,
            debug=self.config.debug,
            tokenizer=tokenizer,
            stop_dict=self.config.stop_dict,
            premise=self.config.premise,
            tokens_column=self.tokens_column,
            ids_column=self.ids_column
        )

        train_df, valid_df = self.pad_and_truncate(
            max_seq=self.config.max_seq,
            train_df=train_df,
            valid_df=valid_df,
            debug=self.config.debug,
            ids_column=self.ids_column,
            len_column=self.len_column,
            mask_column=self.mask_column,
            inf_mask_column=self.inf_mask_column,
            dropped_columns=["id", self.config.premise, self.tokens_column],
            truncate_method=self.config.truncate_method
        )

        columns = train_df.columns.tolist()
        long_field, float_field, label_field, fields = self.prepare_fields(
            columns=columns,
            ids_column=self.ids_column,
            len_column=self.len_column,
            mask_column=self.mask_column,
            inf_mask_column=self.inf_mask_column
        )

        target_obj = (train_df, valid_df, fields, label_field, columns)
        serialize(self.config.dl_path, target_obj)
        return target_obj