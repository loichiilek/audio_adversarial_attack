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

# Constants for the deepspeech model

MODEL_PATH = "./deepspeech-0.6.0-models/output_graph.pbmm"
BEAM_WIDTH = 500

# Audio Path
AUDIO_PATH = "./audio/experience.wav"

# Load pretained DeepSpeech Model
ds = Model(MODEL_PATH, BEAM_WIDTH)

# Read soundfile
audio_data, sr = soundfile.read(AUDIO_PATH ,dtype='int16')

# Deepspeech original interpretation
print("input_word: " + ds.stt(audio_data))

# Specify tokens used
toks = " abcdefghijklmnopqrstuvwxyz'-"

# Define constants
NUM_GEN = 30
NUM_CHILD = 10

# editDistance function for measuring mutated audio's distance from adversarial target
def edit_distance(word1: str, word2: str) -> int:

    memo = {}

    for i in range(len(word1) + 1):
        memo[i] = i

    for y in range(1, len(word2) + 1):
        curr = {}
        curr[0] = y

        for x in range(1, len(word1) + 1):
            if word1[x-1] == word2[y-1]:
                curr[x] = memo[x-1]
            else:
                curr[x] = min([curr[x-1], memo[x], memo[x-1]]) + 1

        memo = curr


    return memo[len(word1)]

def metadata_to_string(metadata):
    return ''.join(item.character for item in metadata.items)

def words_from_metadata(metadata):
    word = ""
    word_list = []
    word_start_time = 0
    # Loop through each character
    for i in range(0, metadata.num_items):
        item = metadata.items[i]
        # Append character to word if it's not a space
        if item.character != " ":
            word = word + item.character
        # Word boundary is either a space or the last character in the array
        if item.character == " " or i == metadata.num_items - 1:
            word_duration = item.start_time - word_start_time

            if word_duration < 0:
                word_duration = 0

            each_word = dict()
            each_word["word"] = word
            each_word["start_time "] = round(word_start_time, 4)
            each_word["duration"] = round(word_duration, 4)

            word_list.append(each_word)
            # Reset
            word = ""
            word_start_time = 0
        else:
            if len(word) == 1:
                # Log the start time of the new word
                word_start_time = item.start_time

    return word_list

def meta_info(metadata):
    word = ""

    for i in range(0, metadata.num_items):
        item = metadata.items[i]
        if item.character == " " or i == metadata.num_items - 1:
            last_word_timestep = int((len(audio_data) / sr - item.start_time) / 0.02)
            for j in range(last_word_timestep):
                word += item.character
            break

        for j in range(metadata.items[i+1].timestep - item.timestep):
            word += item.character
    
    return word

def metadata_json_output(metadata):
    json_result = dict()
    json_result["words"] = words_from_metadata(metadata)
    json_result["confidence"] = metadata.confidence
    return json.dumps(json_result)

def ctc_cost(mutated_word, target_word):
    mutated_vector = tf.convert_to_tensor([toks.index(char) for char in mutated_word])
    target_vector = tf.convert_to_tensor([[toks.index(char) for char in target_word]])
    
    print(target_vector.shape)
    target = ctc_label_dense_to_sparse(target_phrase, target_phrase_lengths)

    tensor, neg_log_prob = tf.nn.ctc_loss(labels=tf.cast(target, tf.int32), inputs=logits, sequence_length=lengths)

    print(tensor)
    print(neg_log_prob)
    print(mutated_vector)
    print(target_vector)

# Set the the target
target_word = "experiment"

original_word = ds.stt(audio_data)
new_audio_data = audio_data
new_word = original_word
cost = edit_distance(original_word, target_word)

# Example of a single mutation in a one generation
#
mutated_word_ex = "eeeeeeeexxxxxxxxxxxxxxxxxxxxxxxxxxxppperrrieeencee"
target_word_ex = "experiment"

mutated_vector = [[toks.index(char) for char in mutated_word_ex]]
target_vector = [[toks.index(char) for char in target_word_ex]]

print(mutated_vector)
print(target_vector)


target_phrase = np.array([list(t)+[0]*(10-len(t)) for t in target_vector])
print(target_phrase)
print(target_phrase.shape)

target_phrase_length = np.array([len(x) for x in target_vector]).astype(np.int32)
print(target_phrase.shape)

target  = ctc_label_dense_to_sparse(target_phrase, target_phrase_length)

print(target)
from tf_logits import get_logits

mutated_tensor = tf.convert_to_tensor(mutated_vector)
mutated_tensor.shape
lengths = [len(mutated_vector)]

logits = get_logits(mutated_tensor, tf.convert_to_tensor(lengths))

print(logits)
ctcloss = tf.nn.ctc_loss(labels=tf.cast(target, tf.int32), inputs=logits, sequence_length=lengths)
print(ctcloss)


with tf.Session() as sess:  
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    print(logits.eval())
    print(ctcloss.eval())
