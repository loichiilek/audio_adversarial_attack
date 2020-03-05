import warnings
warnings.filterwarnings("ignore")


from deepspeech import Model
import wave
import numpy as np
import soundfile
from mutators import *
import scipy.io.wavfile
import json
import tensorflow as tf
import deepspeech
from tensorflow.python.keras.backend import ctc_label_dense_to_sparse

# Specify tokens used
toks = " abcdefghijklmnopqrstuvwxyz'-"

# Define constants
NUM_GEN = 30
NUM_CHILD = 10

class Attack:

    def __init__(self, ds_model, audio_data, sample_rate):
        self.ds_model = ds_model
        self.audio_data = audio_data
        self.sample_rate = sample_rate

    def meta_info(self, metadata):
        word = ""

        for i in range(0, metadata.num_items):
            item = metadata.items[i]
            if item.character == " " or i == metadata.num_items - 1:
                last_word_timestep = int((len(self.audio_data) / self.sample_rate - item.start_time) / 0.02)
                for j in range(last_word_timestep):
                    word += item.character
                break

            for j in range(metadata.items[i+1].timestep - item.timestep):
                word += item.character
        
        return word

    def mutate_audio(self, audio):
        mutated_audio = Mutators.audio_whitenoise(audio, 500).astype(np.int16)
        mutated_metadata = self.ds_model.sttWithMetadata(mutated_audio)
        mutated_word = self.meta_info(mutated_metadata)
        
        return mutated_word

    def attack(self, target_word):
        # Going to do only single attack right now. 
        # However the code of multiple iteration is written.
        mutated_word = self.mutate_audio(self.audio_data)
        mutated_vector = [[toks.index(char) for char in mutated_word]]
        target_vector = [[toks.index(char) for char in target_word]]

        # To create a sparse vector
        target_phrase = np.array([list(t)+[0]*(10-len(t)) for t in target_vector])
        target_phrase_length = np.array([len(x) for x in target_vector]).astype(np.int32)
        target  = ctc_label_dense_to_sparse(target_phrase, target_phrase_length)

        # To get logits
        from tf_logits import get_logits
        mutated_tensor = tf.convert_to_tensor(mutated_vector)
        lengths = [len(mutated_vector)]
        logits = get_logits(mutated_tensor, tf.convert_to_tensor(lengths))

        print(logits)
        ctcloss = tf.nn.ctc_loss(labels=tf.cast(target, tf.int32), inputs=logits, sequence_length=lengths)
        print(ctcloss)

        with tf.Session() as sess:  
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            print(logits.eval())
    # print(ctcloss.eval())


# This is where the main bulk of the code is running
def main():
    # # # # # # # # # # # # # # # # # # # #
    # Constants for the deepspeech model  #
    # # # # # # # # # # # # # # # # # # # #
    MODEL_PATH = "./deepspeech-0.6.0-models/output_graph.pbmm"
    BEAM_WIDTH = 500
    ds = Model(MODEL_PATH, BEAM_WIDTH)


    # Audio Path
    AUDIO_PATH = "./audio/experience.wav"
    # Read soundfile should say experience
    audio_data, sample_rate = soundfile.read(AUDIO_PATH ,dtype='int16')

    # Set the the target
    target_word = "experiment"

    attack = Attack(ds, audio_data, sample_rate)
    attack.attack(target_word)




main()