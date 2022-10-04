from cgi import test
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
train_epochs = 5

train_file = F'{squad_dir}/train-{squad_version}.json'
dev_file = F'{squad_dir}/dev-{squad_version}.json'

tokenizer = nlp.layers.FastWordpieceBertTokenizer(
    vocab_file=F'{bert_dir}/vocab.txt', lower_case=True)

with tf.io.gfile.GFile(train_file, "r") as reader:
    input_data = json.load(reader)["data"]


def improve_answer_span(doc_tokens, input_start, input_end, tokenizer, orig_answer_text):

    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False


def print_output(name, output):
    print(F'--------------{name}-----------------')
    print(output)


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
        'input_mask': example["input_mask"],
        'is_impossible': example["is_impossible"]
    }, {
        "start_positions": example["start_positions"],
        "end_positions": example["end_positions"],
    }


# for raw_record in raw_dataset.take(10).map(decode_fn):
#     print(raw_record)

bert_config_file = os.path.join(bert_dir, 'bert_config.json')
config_dict = json.loads(tf.io.gfile.GFile(bert_config_file).read())

dataset = raw_dataset.map(decode_fn)

dataset = dataset.filter(lambda x, y: x['is_impossible'] == 0)

train_ds = dataset.take(2000).batch(batch_size=train_batch_size)

test_ds = dataset.skip(2000).take(500).batch(batch_size=train_batch_size)

td = list(raw_dataset.take(1).map(decode_fn).as_numpy_iterator())
print(td)

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


bert_span.compile(optimizer='adam',
                  loss={
                      'start_positions': 'sparse_categorical_crossentropy',
                      'end_positions': 'sparse_categorical_crossentropy'
                  },
                  metrics=['accuracy'])

# bert_span.evaluate(test_ds)

bert_span.fit(train_ds,
              validation_data=(test_ds),
              epochs=train_epochs)
