from cgi import test
import os
import numpy as np
import tensorflow as tf
from tensorflow_models import nlp, optimization
from keras import metrics, layers, optimizers, losses, Model, models, callbacks
from keras.utils import plot_model
import json

checkpoint_dir = "./model_save"
database_dir = "/home/pjb/Documents/github/datas"
bert_dir = database_dir+"/bert_base"
squad_version = "v2.0"
squad_dir = database_dir+"/squad"
max_seq_length = 384
train_batch_size = 5
train_epochs = 5


tokenizer = nlp.layers.FastWordpieceBertTokenizer(
    vocab_file=F'{bert_dir}/vocab.txt', lower_case=True)

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


bert_config_file = os.path.join(bert_dir, 'bert_config.json')

config_dict = json.loads(tf.io.gfile.GFile(bert_config_file).read())

dataset = raw_dataset.shuffle(buffer_size=train_batch_size).map(decode_fn)

# 总条目131944,可用条目87212
train_num = 30


train_ds = dataset.take(train_num).batch(batch_size=train_batch_size)

test_ds = dataset.skip(train_num).take(train_batch_size).batch(batch_size=train_batch_size)


def get_optimizer():
    train_data_size = train_num
    steps_per_epoch = int(train_data_size / train_batch_size)
    num_train_steps = steps_per_epoch * train_epochs
    warmup_steps = int(0.1 * num_train_steps)
    initial_learning_rate = 2e-5

    linear_decay = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=initial_learning_rate,
        end_learning_rate=0,
        decay_steps=num_train_steps)

    warmup_schedule = optimization.lr_schedule.LinearWarmup(
        warmup_learning_rate=0,
        after_warmup_lr_sched=linear_decay,
        warmup_steps=warmup_steps
    )

    optimizer = optimizers.Adam(
        learning_rate=warmup_schedule)

    return optimizer


def build_model():
    encoder_config = nlp.encoders.EncoderConfig({
        'type': 'bert',
        'bert': config_dict
    })

    bert_encoder = nlp.encoders.build_encoder(encoder_config)
    #bert_encoder.trainable = False
    # output='predictions'
    bert_span = nlp.models.BertSpanLabeler(network=bert_encoder)

    checkpoint = tf.train.Checkpoint(encoder=bert_encoder)

    checkpoint.read(
        os.path.join(bert_dir, 'bert_model.ckpt')
    ).assert_consumed()

    return bert_span


def make_or_restore_model():

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    checkpoints = [checkpoint_dir + "/" +
                   name for name in os.listdir(checkpoint_dir)]
    if checkpoints:

        latest_checkpoint = max(checkpoints, key=os.path.getctime)

        print("Restoring from", latest_checkpoint)

        model = models.load_model(latest_checkpoint, compile=False)
    else:
        print("Creating a new model")
        model = build_model()

    metrics = [tf.keras.metrics.SparseCategoricalAccuracy(
        'accuracy', dtype=tf.float32)]

    loss = [losses.SparseCategoricalCrossentropy(from_logits=True),
            losses.SparseCategoricalCrossentropy(from_logits=True)]

    optimizer = get_optimizer()

    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metrics)

    return model


model = make_or_restore_model()

callbacks = [
    callbacks.ModelCheckpoint(
        filepath=checkpoint_dir + "/ckpt-{epoch:02d}-{val_loss:.2f}",
        save_best_only=True,
        save_freq="epoch",
        verbose=1
    )
]

# model.evaluate(test_ds)

model.fit(train_ds,
          validation_data=(test_ds),
          batch_size=train_batch_size,
          epochs=train_epochs,
          callbacks=callbacks)

models.saved_model(model, "./qa_model")
