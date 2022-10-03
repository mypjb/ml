import os
import numpy as np
import tensorflow as tf
from tensorflow_models import nlp
from keras import metrics, layers, optimizers, losses, Model, Sequential, Input
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

packer = nlp.layers.BertPackInputs(seq_length=max_seq_length,
                                   special_tokens_dict=tokenizer.get_special_tokens_dict())


class BertInputProcessor(layers.Layer):
    def __init__(self,
                 tokenizer, packer,
                 paragraph_name='context',
                 question_name='question',
                 label_name='label'):
        super().__init__()
        self.tokenizer = tokenizer
        self.packer = packer
        self.paragraph_name = paragraph_name
        self.question_name = question_name
        self.label_name = label_name

    def call(self, inputs):

        paragraph_text = inputs[self.paragraph_name]
        paragraph_tokens = self.tokenizer(tf.constant([paragraph_text]))
        question_tokens = self.tokenizer(
            tf.constant([inputs[self.question_name]]))

        packed = self.packer([paragraph_tokens, question_tokens])

        if self.label_name in inputs:
            label = inputs[self.label_name]

            return { **packed, **label }
        else:
            return packed


def print_output(name, output):
    print(F'--------------{name}-----------------')
    print(output)


def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


def raw_read():
    dataset = []
    for entry in input_data:
        for paragraph in entry["paragraphs"]:
            paragraph_text = paragraph["context"]
            doc_tokens = []
            char_to_word_offset = []
            prev_is_whitespace = True
            for c in paragraph_text:
                if is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        doc_tokens.append(c)
                    else:
                        doc_tokens[-1] += c
                    prev_is_whitespace = False
                char_to_word_offset.append(len(doc_tokens) - 1)

            for qa in paragraph["qas"]:

                if qa["is_impossible"]:
                    continue
                question_text = qa["question"]
                start_position = None
                end_position = None
                orig_answer_text = None
                answer = qa["answers"][0]
                orig_answer_text = answer["text"]
                answer_offset = answer["answer_start"]

                answer_length = len(orig_answer_text)
                start_position = char_to_word_offset[answer_offset]
                end_position = char_to_word_offset[answer_offset +
                                                   answer_length - 1]

                input = {
                    "context": paragraph_text,
                    "question": question_text,
                    "label": {
                        'start_position': start_position,
                        'end_position': end_position
                    }
                }

                dataset.append(input)
                if len(dataset) >= 1100:
                    break
            if len(dataset) >= 1100:
                break
        if len(dataset) >= 1100:
            break

    return dataset


bert_input_processor = BertInputProcessor(tokenizer=tokenizer, packer=packer)

datasets = raw_read()

train_ds = np.array([bert_input_processor(x) for x in datasets[0:5]])
test_ds = [bert_input_processor(x) for x in datasets[100:105]]

# for raw_record in raw_dataset.take(10).map(decode_fn):
#     print(raw_record)

bert_config_file = os.path.join(bert_dir, 'bert_config.json')

config_dict = json.loads(tf.io.gfile.GFile(bert_config_file).read())


encoder_config = nlp.encoders.EncoderConfig({
    'type': 'bert',
    'bert': config_dict
})

bert_encoder = nlp.encoders.build_encoder(encoder_config)

model = nlp.models.BertSpanLabeler(network=bert_encoder)

plot_model(model, show_shapes=True, to_file="bert.png")

checkpoint = tf.train.Checkpoint(encoder=bert_encoder)

checkpoint.read(
    os.path.join(bert_dir, 'bert_model.ckpt')
).assert_consumed()


model.compile(optimizer='adam',
              loss={
                  'start_position': 'sparse_categorical_crossentropy',
                  'end_position': 'sparse_categorical_crossentropy'
              },
              metrics=['accuracy'])

model.evaluate(tf.constant(test_ds))

model.fit(train_ds,
          validation_data=(test_ds),
          batch_size=train_batch_size,
          epochs=train_epochs)
