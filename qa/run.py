import os
import numpy as np
import tensorflow as tf
import tensorflow_models
from tensorflow_models import nlp
from keras import metrics, layers, optimizers, losses, Model, Sequential
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

class SquadExample(object):
    def __init__(self, context,
                 question,
                 start_position,
                 end_position,
                 is_impossible):
        self.context = context
        self.question = question
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible


def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False


squad_trains = []

for items in input_data[0:1000]:
    for paragraph in items['paragraphs']:
        paragraph_text = paragraph['context']
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

        for qa in paragraph['qas']:

            question_text = qa["question"]
            is_impossible = qa['is_impossible']

            if is_impossible:
                start_position = -1
                end_position = -1
            else:
                
                answer = qa["answers"][0]
                answer_length = len(answer)
                answer_offset = answer["answer_start"]

                if (len(char_to_word_offset) <= answer_offset+answer_length-1):
                    continue

                start_position = char_to_word_offset[answer_offset]
                end_position = char_to_word_offset[answer_offset +
                                                   answer_length - 1]

            squad_trains.append(SquadExample(paragraph_text,
                                             question_text,
                                             start_position,
                                             end_position,
                                             is_impossible))


packer = nlp.layers.BertPackInputs(seq_length=max_seq_length,
                                   special_tokens_dict=tokenizer.get_special_tokens_dict())


class BertInputProcessor(tf.keras.layers.Layer):
    def __init__(self, tokenizer, packer):
        super().__init__()
        self.tokenizer = tokenizer
        self.packer = packer

    def call(self, inputs):
        context_token = self.tokenizer(inputs['context'])
        question_token = self.tokenizer(inputs['question'])

        packed = self.packer([context_token, question_token])
        labels = {
            'start_positions': inputs['start_position'], 'end_positions': inputs['end_position']}
        return (packed, labels)


bert_input_processor = BertInputProcessor(tokenizer, packer)


def data_preprocessing(datas):

    data_dict = {
        'context': [],
        'question': [],
        'start_position': [],
        'end_position': []
    }

    for data in datas:
        data_dict['context'].append(data.context)
        data_dict['question'].append(data.question)
        data_dict['start_position'].append(data.end_position)
        data_dict['end_position'].append(data.end_position)

    dataset = bert_input_processor(data_dict)
    return tf.data.Dataset.from_tensor_slices(dataset)

train_ds = data_preprocessing(squad_trains[0:1000]).batch(batch_size=train_batch_size)
test_ds = data_preprocessing(squad_trains[1000:1300]).batch(batch_size=train_batch_size)


bert_config_file = os.path.join(bert_dir, 'bert_config.json')

config_dict = json.loads(tf.io.gfile.GFile(bert_config_file).read())


encoder_config = nlp.encoders.EncoderConfig({
    'type': 'bert',
    'bert': config_dict
})

bert_encoder = nlp.encoders.build_encoder(encoder_config)

model = nlp.models.BertSpanLabeler(network=bert_encoder)

plot_model(model, show_shapes=True, expand_nested=True,to_file="bert.png")

checkpoint = tf.train.Checkpoint(model=bert_encoder, encoder=bert_encoder)

checkpoint.read(
    os.path.join(bert_dir, 'bert_model.ckpt')
).assert_consumed()



model.compile(optimizer='adam',
              loss={"start_positions": "sparse_categorical_crossentropy",
                    "end_positions": "sparse_categorical_crossentropy"},
              metrics=['accuracy'])

# model.evaluate(tf.constant(test_ds))

model.fit(train_ds,
          validation_data=(test_ds),
          batch_size=train_batch_size,
          epochs=train_epochs)
