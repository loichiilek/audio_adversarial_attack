## attack.py -- generate audio adversarial examples
##
## Copyright (C) 2017, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

from deepspeech import Model
import numpy as np
import tensorflow as tf
import argparse
import scipy.io.wavfile as wav
import os
import sys

sys.path.append("DeepSpeech")

import DeepSpeech

from tensorflow.python.keras.backend import ctc_label_dense_to_sparse
from tf_logits import get_logits

# These are the tokens that we're allowed to use.
# The - token is special and corresponds to the epsilon
# value in CTC decoding, and can not occur in the phrase.
toks = " abcdefghijklmnopqrstuvwxyz'-"


class Attack:
    def __init__(self, sess, phrase_length, max_audio_len, batch_size=1, model_path=None):
        """
        Set up the attack procedure.

        Here we create the TF graph that we're going to use to
        actually generate the adversarial examples.
        """
        restore_path = model_path + "/model.v0.4.1"
        self.sess = sess
        self.ds = self.build_model(model_path)
        self.batch_size = batch_size
        self.phrase_length = phrase_length
        self.max_audio_len = max_audio_len

        # Create all the variables necessary
        # they are prefixed with qq_ just so that we know which
        # ones are ours so when we restore the session we don't
        # clobber them.
        # self.delta = delta = tf.Variable(np.zeros((batch_size, max_audio_len), dtype=np.float32), name='qq_delta')
        self.mask = mask = tf.Variable(np.zeros((batch_size, max_audio_len), dtype=np.float32), name='qq_mask')
        self.cwmask = cwmask = tf.Variable(np.zeros((batch_size, phrase_length), dtype=np.float32), name='qq_cwmask')
        self.original = original = tf.Variable(np.zeros((batch_size, max_audio_len), dtype=np.float32), name='qq_original')
        self.lengths = lengths = tf.Variable(np.zeros(batch_size, dtype=np.int32), name='qq_lengths')
        self.target_phrase = tf.Variable(np.zeros((batch_size, phrase_length), dtype=np.int32), name='qq_phrase')
        self.target_phrase_lengths = tf.Variable(np.zeros((batch_size), dtype=np.int32), name='qq_phrase_lengths')
        

        # We set the new input to the model to be the abve delta
        # plus a mask, which allows us to enforce that certain
        # values remain constant 0 for length padding sequences.
        self.new_input = new_input = mask + original

        # We add a tiny bit of noise to help make sure that we can
        # clip our values to 16-bit integers and not break things.
        noise = tf.random_normal(new_input.shape, stddev=2)
        pass_in = tf.clip_by_value(new_input + noise, -2 ** 15, 2 ** 15 - 1)

        # Feed this final value to get the logits.
        self.logits = logits = get_logits(pass_in, lengths)

        # And finally restore the graph to make the classifier
        # actually do something interesting.
        saver = tf.train.Saver([x for x in tf.global_variables() if 'qq' not in x.name])
        saver.restore(sess, restore_path)

        target = ctc_label_dense_to_sparse(self.target_phrase, self.target_phrase_lengths)

        ctcloss = tf.nn.ctc_loss(labels=tf.cast(target, tf.int32),
                                    inputs=logits, sequence_length=lengths)

        self.expanded_loss = tf.constant(0)
        self.ctcloss = ctcloss

        # Decoder from the logits, to see how we're doing
        self.decoded, _ = tf.nn.ctc_beam_search_decoder(logits, lengths, merge_repeated=False, beam_width=100)

    def build_model(self, model_path):

        # Build model 
        BEAM_WIDTH = 500
        LM_ALPHA = 0.75
        LM_BETA = 1.85
        N_FEATURES = 26
        N_CONTEXT = 9
        MODEL_PATH = model_path + '/models/output_graph.pb'
        ALPHABET_PATH = model_path + '/models/alphabet.txt'
        LM_PATH = model_path + '/models/lm.binary'
        TRIE_PATH = model_path + '/models/trie'

        ds = Model(MODEL_PATH, N_FEATURES, N_CONTEXT, ALPHABET_PATH, BEAM_WIDTH)
        ds.enableDecoderWithLM(ALPHABET_PATH, LM_PATH, TRIE_PATH, LM_ALPHA, LM_BETA)

        return ds

    def attack(self, audio, lengths, target, print_toggle):
        sess = self.sess

        # Initialize all of the variables
        # TODO: each of these assign ops creates a new TF graph
        # object, and they should be all created only once in the
        # constructor. It works fine as long as you don't call
        # attack() a bunch of times.
        # sess.run(tf.variables_initializer([self.delta]))
        sess.run(self.original.assign(np.array(audio)))
        sess.run(self.lengths.assign((np.array(lengths) - 1) // 320))
        sess.run(self.mask.assign(np.array([[1 if i < l else 0 for i in range(self.max_audio_len)] for l in lengths])))
        sess.run(self.cwmask.assign(
            np.array([[1 if i < l else 0 for i in range(self.phrase_length)] for l in (np.array(lengths) - 1) // 320])))
        sess.run(self.target_phrase_lengths.assign(np.array([len(x) for x in target])))
        sess.run(self.target_phrase.assign(np.array([list(t) + [0] * (self.phrase_length - len(t)) for t in target])))

        # We'll make a bunch of iterations of gradient descent here
        el, cl, logits, new_input = sess.run((self.expanded_loss, self.ctcloss, self.logits, self.new_input))




        # Crossover parents
        optimal_cost1 = max(cl[0], cl[1])
        optimal_cost2 = min(cl[0], cl[1])

        for i in range(2, len(cl)):
            if cl[i] < optimal_cost1:
                optimal_cost2 = optimal_cost1
                optimal_cost1 = cl[i]

            elif cl[i] < optimal_cost2:
                optimal_cost2 = cl[i]

        optimal_index1 = np.where(cl == optimal_cost1)
        optimal_index2 = np.where(cl == optimal_cost2)

        # Find optimal index
        optimal_cost = cl[optimal_index1]
        optimal_audio1 = new_input[optimal_index1]
        optimal_audio2 = new_input[optimal_index2]


        if print_toggle:
            # Try to retrieve the decoded words
            out = sess.run(self.decoded)

            res = np.zeros(out[0].dense_shape)+len(toks)-1
        
            for ii in range(len(out[0].values)):
                x,y = out[0].indices[ii]
                res[x,y] = out[0].values[ii]

            res = ["".join(toks[int(x)] for x in y) for y in res]
            print("======================================================")
            print("Current decoded word: " + self.ds.stt(optimal_audio1[0].astype(np.int16), 16000))
            print("Current decoded word (without language model): " + str(res[optimal_index1[0][0]]))
            print("Average loss: %.3f" % np.mean(cl) + "\n")

        return optimal_cost[0], optimal_audio1[0].astype(np.int16), optimal_audio2[0].astype(np.int16)

def mutate_audio(audio, num, mutation_range):

    audios = []
    lengths = []
    for i in range(num):
        wn = np.random.randint(-mutation_range, mutation_range, size=len(audio), dtype=np.int16)
        mutated_audio = audio + wn
        audios.append(list(mutated_audio))
        lengths.append(len(mutated_audio))

    return audios, lengths

# Perform single point crossover on the audio parents
def crossover_audio(audio1, audio2, num_children):

    audios = []
    lengths = []

    for i in range(num_children):
        crossover_point = np.random.randint(0, len(audio1))

        assert len(audio1) == len(audio2)
        child1 = np.concatenate((audio1[:crossover_point], audio2[crossover_point:]))
        child2 = np.concatenate((audio2[:crossover_point], audio1[crossover_point:]))

        audios.append(list(child1))
        audios.append(list(child2))
        lengths.append(len(child1))
        lengths.append(len(child2))
    
    return audios, lengths 



def main():
    """
    Do the attack here.

    This is all just boilerplate; nothing interesting
    happens in this method.

    For now we only support using CTC loss and only generating
    one adversarial example at a time.
    """

    mutation_range = 150
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--input', type=str, dest="input",
                        required=True,
                        help="Input audio .wav file(s) at 16KHz")
    parser.add_argument('--target', type=str,
                        required=True,
                        help="Target transcription")
    parser.add_argument('--out', type=str,
                        required=True,
                        help="Path for the adversarial example(s)")
    parser.add_argument('--iterations', type=int,
                        required=False, default=1000,
                        help="Maximum number of iterations of gradient descent")
    parser.add_argument('--population', type=int,
                        required=False, default=100,
                        help="Population size of each generation")
    parser.add_argument('--model_path', type=str,
                        required=True,
                        help="Path to the DeepSpeech checkpoint (ending in model0.4.1)")
    args = parser.parse_args()
    while len(sys.argv) > 1:
        sys.argv.pop()

    population_size = args.population

    with tf.Session() as sess:
        # finetune = []
        audios = []
        lengths = []

        # if args.out is None:
        #     assert args.outprefix is not None
        # else:
        #     assert args.outprefix is None
        #     assert len(args.input) == len(args.out)
        # if args.finetune is not None and len(args.finetune):
        #     assert len(args.input) == len(args.finetune)

        # Load the inputs that we're given
        
        fs, audio = wav.read(args.input)
        # print("Original Audio: " + interpret_audio(audio, fs))
        assert fs == 16000
        assert audio.dtype == np.int16
        # print(audio)
        # print('source dB', 20 * np.log10(np.max(np.abs(audio))))
        audios.append(list(audio))
        lengths.append(len(audio))

        for i in range(population_size):
            wn = np.random.randint(-mutation_range, mutation_range, size=len(audio), dtype=np.int16)
            mutated_audio = audio + wn
            audios.append(list(mutated_audio))
            lengths.append(len(mutated_audio))

            # if args.finetune is not None:
            #     finetune.append(list(wav.read(args.finetune[i])[1]))


        maxlen = max(map(len, audios))
        audios = np.array([x + [0] * (maxlen - len(x)) for x in audios])

        phrase = args.target
        # Set up the attack class and run it
        attack = Attack(sess, len(phrase), maxlen, batch_size=len(audios), model_path=args.model_path)
        
        

        optimal_cost, optimal_audio1, optimal_audio2 = attack.attack(audios, lengths, [[toks.index(x) for x in phrase]] * len(audios), True)
        crossover_population = int(0.2*population_size)
        mutation_population = population_size - (2 * crossover_population)

        for i in range(args.iterations):
            # Reset audios to only the generational best audio
            print_toggle = False
            if (i+1) % 10 == 0:
                print_toggle = True

            audios = [optimal_audio1]
            lengths = [len(optimal_audio1)]

            

            mutated_audios, mutated_lengths = mutate_audio(optimal_audio1, mutation_population, mutation_range)
            crossover_audios, crossover_lengths = crossover_audio(optimal_audio1, optimal_audio2, crossover_population)

            audios.extend(mutated_audios)
            audios.extend(crossover_audios)

            lengths.extend(mutated_lengths)
            lengths.extend(crossover_lengths)

            
            xcost, xaudio1, xaudio2 = attack.attack(audios, lengths, [[toks.index(x) for x in phrase]] * len(audios), print_toggle)

            if xcost < optimal_cost:
                optimal_cost = xcost
                optimal_audio1 = xaudio1
                optimal_audio2 = xaudio2
            
            print("iteration: " + str(i+1) + "\t" + "Cost: " + str(optimal_cost))

        wav.write(args.out, 16000, optimal_audio1)
        
main()
