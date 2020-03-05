import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import tensorflow as tf

# test

def compute_mfcc(audio, **kwargs):
    """
    Compute the MFCC for a given audio waveform. This is
    identical to how DeepSpeech does it, but does it all in
    TensorFlow so that we can differentiate through it.
    """

    batch_size, size = audio.get_shape().as_list()
    print("batch size: " + str(batch_size))
    print("size: " + str(size))

    audio = tf.cast(audio, tf.float32)

    # 1. Pre-emphasizer, a high-pass filter
    audio = tf.concat((audio[:, :1], audio[:, 1:] - 0.97*audio[:, :-1], np.zeros((batch_size,512),dtype=np.float32)), 1)
    
    # with tf.Session() as sess:  print(audio.eval())
    print(audio.shape)
    for i in range(0, size-320, 320):
        print("i: " + str(i))

    print([i for i in range(0,size-10 ,10)])
    # 2. windowing into frames of 512 samples, overlapping
    windowed = tf.stack([audio[:, i:i+512] for i in range(0,size-10 ,10)],1)

    window = np.hamming(512)
    windowed = windowed * window

    # 3. Take the FFT to convert to frequency space
    ffted = tf.signal.rfft(windowed, [512])
    ffted = 1.0 / 512 * tf.square(tf.abs(ffted))

    # 4. Compute the Mel windowing of the FFT
    energy = tf.reduce_sum(ffted,axis=2)+np.finfo(float).eps
    filters = np.load("filterbanks.npy").T
    feat = tf.matmul(ffted, np.array([filters]*batch_size,dtype=np.float32))+np.finfo(float).eps

    # 5. Take the DCT again, because why not
    feat = tf.math.log(feat)
    feat = tf.signal.dct(feat, type=2, norm='ortho')[:,:,:26]

    # 6. Amplify high frequencies for some reason
    _,nframes,ncoeff = feat.get_shape().as_list()
    n = np.arange(ncoeff)
    lift = 1 + (22/2.)*np.sin(np.pi*n/22)
    feat = lift*feat
    width = feat.get_shape().as_list()[1]


    # 7. And now stick the energy next to the features
    feat = tf.concat((tf.reshape(tf.math.log(energy),(-1,width,1)), feat[:, :, 1:]), axis=2)

    return feat


toks = " abcdefghijklmnopqrstuvwxyz'-"
mutated_word_ex = "eeeeeeeexxxxxxxxxxxxxxxxxxxxxxxxxxxppperrrieeencee"
mutated_vector = [[toks.index(char) for char in mutated_word_ex]]
mutated_tensor = tf.convert_to_tensor(mutated_vector)

feat = compute_mfcc(mutated_tensor)

print(feat)


import time
import os
import sys
sys.path.append("DeepSpeech")
import DeepSpeech
def get_logits(new_input, length, first=[]):
    """
    Compute the logits for a given waveform.

    First, preprocess with the TF version of MFC above,
    and then call DeepSpeech on the features.
    """

    batch_size = new_input.get_shape()[0]

    # 1. Compute the MFCCs for the input audio
    # (this is differentable with our implementation above)
    empty_context = np.zeros((batch_size, 9, 26), dtype=np.float32)
    new_input_to_mfcc = compute_mfcc(new_input)
    features = tf.concat((empty_context, new_input_to_mfcc, empty_context), 1)

    # 2. We get to see 9 frames at a time to make our decision,
    # so concatenate them together.
    features = tf.reshape(features, [new_input.get_shape()[0], -1])
    features = tf.stack([features[:, i:i+19*26] for i in range(0,features.shape[1]-19*26+1,26)],1)
    features = tf.reshape(features, [batch_size, -1, 19, 26])


    # 3. Finally we process it with DeepSpeech
    # We need to init DeepSpeech the first time we're called
    if first == []:
        first.append(False)

        DeepSpeech.create_flags()
        tf.app.flags.FLAGS.alphabet_config_path = "DeepSpeech/data/alphabet.txt"
        DeepSpeech.initialize_globals()

    logits, _ = DeepSpeech.BiRNN(features, length, [0]*10)