import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import tensorflow as tf
import argparse
import soundfile
import os
import sys
from deepspeech import Model

from tensorflow.python.keras.backend import ctc_label_dense_to_sparse
from tf_logits import get_logits

toks = " abcdefghijklmnopqrstuvwxyz'-"


class Attack:
    def __init__(self, sess, ds_model, batch_size, max_audio_len):
        self.sess = sess
        self.ds_model = ds_model
        self.batch_size = batch_size
        
        # Creating necessary variables
        self.original = original = tf.Variable(np.zeros((batch_size, max_audio_len), dtype=np.float32), name='qq_original')
        self.lengths = lengths = tf.Variable(np.zeros(batch_size, dtype=np.int32), name='qq_lengths')
        self.target_phrase = tf.Variable(np.zeros((batch_size, 1), dtype=np.int32), name='qq_phrase')
        self.target_phrase_length = tf.Variable(np.zeros((batch_size), dtype=np.int32), name='qq_phrase_length')

        
        noise = tf.random_normal(original.shape, stddev=2)
        pass_in = tf.clip_by_value(original+noise, -2**15, 2**15-1)
        self.logits = logits = get_logits(pass_in, lengths)

        
        target  = ctc_label_dense_to_sparse(self.target_phrase, self.target_phrase_length)

        ctcloss = tf.nn.ctc_loss(labels=tf.cast(target, tf.int32),
                                     inputs=logits, sequence_length=lengths)

    def attack(self, audios, lengths, target):
        sess = self.sess
        original = self.original

        # To create a sparse vector
        target_vector = [[toks.index(char) for char in target]]
        # target_phrase = np.array([list(t)+[0]*(10-len(t)) for t in target_vector])
        # target_phrase_length = np.array([len(x) for x in target_vector]).astype(np.int32)

        # To create logits
        sess.run(self.original.assign(np.array(audios)))
        sess.run(self.lengths.assign((np.array(lengths)-1)//320))
        sess.run(self.target_phrase_lengths.assign(np.array([len(x) for x in target_vector]).astype(np.int32)))
        sess.run(self.target_phrase.assign(np.array([list(t)+[0]*(10-len(t)) for t in target_vector])))


        print(self.original.eval())
        
        


def main():

    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--input', type=str, dest="input", nargs='+', 
                        required=True, 
                        help="Input audio .wav file(s), at 16KHz (separated by spaces)")

    parser.add_argument('--target', type=str,
                        required=True,
                        help="Target transcription")

    
    parser.add_argument('--model', type=str,
                        required=True,
                        help="Model path")

    args = parser.parse_args()
    while len(sys.argv) > 1:
        sys.argv.pop()
    
    MODEL_PATH = os.path.join(args.model, 'output_graph.pb')
    ALPHABET_PATH = os.path.join(args.model, 'alphabet.txt')
    BEAM_WIDTH = 500
    LM_ALPHA = 0.75
    LM_BETA = 1.85
    N_FEATURES = 26
    N_CONTEXT = 9

    with tf.compat.v1.Session() as sess:
        audios = []
        lengths = []

        for i in range(len(args.input)):
            audio_data, sample_rate = soundfile.read(args.input[i] ,dtype='int16') 
            assert sample_rate == 16000

            audios.append(list(audio_data))
            lengths.append(len(audio_data))


        max_audio_len = max(lengths)
        audios = np.array([x+[0]*(max_audio_len-len(x)) for x in audios])

        target_phrase = args.target
        ds = Model(MODEL_PATH, N_FEATURES, N_CONTEXT, ALPHABET_PATH, BEAM_WIDTH)
        batch_size = len(audios)


        attack = Attack(sess, ds, batch_size, max_audio_len)
        attack.attack(audios, lengths, "test")



main()