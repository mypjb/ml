import os
import numpy as np
import tensorflow as tf
from tensorflow_models import nlp
from keras import metrics, layers, optimizers, losses, Model
from keras.utils import plot_model
import json

database_dir = "/home/pjb/Documents/github/datas"
bert_dir = database_dir+"/bert_base"
squad_version = "v2.0"
squad_dir = database_dir+"/squad"
max_seq_length = 384
train_batch_size = 10

raw_dataset = tf.data.TFRecordDataset([os.path.join(
    squad_dir, f"train_{squad_version}.tf_record")])

feature_description = {
    'unique_ids': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'input_ids': tf.io.FixedLenFeature(shape=(max_seq_length,), dtype=tf.int64, default_value=[0] * max_seq_length),
    'input_mask': tf.io.FixedLenFeature(shape=(max_seq_length,), dtype=tf.int64, default_value=[0] * max_seq_length),
    'segment_ids': tf.io.FixedLenFeature(shape=(max_seq_length,), dtype=tf.int64, default_value=[0] * max_seq_length),
    'paragraph_mask': tf.io.FixedLenFeature(shape=(max_seq_length,), dtype=tf.int64, default_value=[0] * max_seq_length),
    'class_index': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'start_positions': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'end_positions': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'is_impossible': tf.io.FixedLenFeature([], tf.int64, default_value=0)
}


def decode_fn(record_bytes):
    example = tf.io.parse_single_example(
        record_bytes,
        feature_description
    )
    # example["input_word_ids"] = example["input_ids"]
    # example["input_type_ids"] = example["segment_ids"]

    return {
        'input_word_ids': example["input_ids"],
        'input_type_ids': example["segment_ids"],
        'input_mask': example["input_mask"]
    }, {
        "start_positions": example["start_positions"],
        "end_positions": example["end_positions"],
    }


# for raw_record in raw_dataset.take(10).map(decode_fn):
#     print(raw_record)

bert_config_file = os.path.join(bert_dir, 'bert_config.json')
config_dict = json.loads(tf.io.gfile.GFile(bert_config_file).read())

dataset = raw_dataset.shuffle(buffer_size=train_batch_size).map(decode_fn)


dataset = dataset.batch(
    batch_size=train_batch_size)

# print(dataset.take(2))

# exit(0)
train_ds = dataset.take(2000)

test_ds = dataset.skip(2000).take(1000)

# print(list(train_ds.take(1).as_numpy_iterator()))

# print(config_dict)

# tokenizer = nlp.layers.FastWordpieceBertTokenizer(
#     vocab_file=os.path.join(bert_dir, "vocab.txt"),
#     lower_case=True)


# packer = nlp.layers.BertPackInputs(
#     seq_length=max_seq_length,
#     special_tokens_dict=tokenizer.get_special_tokens_dict()
# )


# class BertInputProcessor(layers.Layer):
#     def __init__(self, tokenizer, packer):
#         super().__init__()
#         self.tokenizer = tokenizer
#         self.packer = packer

#     def call(self, inputs):
#         tok1 = self.tokenizer(inputs['sentence1'])
#         tok2 = self.tokenizer(inputs['sentence2'])

#         packed = self.packer([tok1, tok2])

#         if 'label' in inputs:
#             return packed, inputs['label']
#         else:
#             return packed


# bert_input_processor = BertInputProcessor(tokenizer, packer)


encoder_config = nlp.encoders.EncoderConfig({
    'type': 'bert',
    'bert': config_dict
})

bert_encoder = nlp.encoders.build_encoder(encoder_config)

bert_span = nlp.models.BertSpanLabeler(network=bert_encoder)


plot_model(bert_span, show_shapes=True, to_file="bert.png")

checkpoint = tf.train.Checkpoint(encoder=bert_encoder)

checkpoint.read(
    os.path.join(bert_dir, 'bert_model.ckpt')
).assert_consumed()

# metrics = [metrics.SparseCategoricalAccuracy('accuracy')]

# loss = losses.SparseCategoricalCrossentropy(from_logits=True)

# optimizer = optimizers.Adam()

bert_span.compile(optimizer='adam',
                  loss={
                      'start_positions': 'sparse_categorical_crossentropy',
                      'end_positions': 'sparse_categorical_crossentropy'
                  },
                  metrics=['accuracy'])

# bert_span.evaluate(test_ds)

bert_span.fit(train_ds,
              validation_data=(test_ds),
              batch_size=train_batch_size,
              epochs=5)
