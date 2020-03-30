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
    def __init__(self, sess, phrase_length, max_audio_len, batch_size=1,
                  restore_path=None):
        """
        Set up the attack procedure.

        Here we create the TF graph that we're going to use to
        actually generate the adversarial examples.
        """

        self.sess = sess
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

        if print_toggle:
            optimal_index = np.where(cl == min(cl))
            optimal_cost = cl[optimal_index]
            optimal_audio = new_input[optimal_index]
            # Try to retrieve the decoded words
            out = sess.run(self.decoded)

            res = np.zeros(out[0].dense_shape)+len(toks)-1
        
            for ii in range(len(out[0].values)):
                x,y = out[0].indices[ii]
                res[x,y] = out[0].values[ii]

            res = ["".join(toks[int(x)] for x in y) for y in res]
            print("======================================================")
            print("Current decoded word (without language model): " + str(res[optimal_index[0][0]]))
            print("Average loss: %.3f" % np.mean(cl) + "\n")

        # return new audios and new cost
        return new_input, cl

def mutate_audio(audio, num, mutation_range):

    audios = []
    lengths = []


    for i in range(num):
        wn = np.random.randint(-mutation_range, mutation_range, size=len(audio), dtype=np.int16)
        mutated_audio = audio + wn
        audios.append(list(mutated_audio))
        lengths.append(len(mutated_audio))

    return audios, lengths

# Create the particle swarming class
class PSOEnvironment():
    '''
    The hyperparameters set in this init function such as w, c1 and c2 
    are predetermined from other research, which conclude that these 
    values are optimal.
    '''
    def __init__(self, num_particle, audio, model_path, target, sess):
        
        self.global_min_cost = float('inf')
        self.gbest_position = audio
        self.target = target
        self.lengths = []
        self.w = 0.9
        self.c1 = 2.05
        self.c2 = 2.05
        self.ds = self.build_model(model_path)
        restore_path = model_path + "/model.v0.4.1"

        audios = []

        # To create the first set of particles here
        # Creating first set of mutation
        for _ in range(num_particle):
            wn = np.random.randint(-200, 200, size=len(audio), dtype=np.int16)
            mutated_audio = audio + wn
            audios.append(list(mutated_audio))
            self.lengths.append(len(mutated_audio))

        maxlen = max(map(len, audios))
        audios = np.array([x + [0] * (maxlen - len(x)) for x in audios])

        self.attack = Attack(sess, len(target), maxlen, batch_size=len(audios), restore_path=restore_path)
        new_input, cl = self.attack.attack(audios, self.lengths, [[toks.index(x) for x in self.target]] * len(audios), True)

        # Instantiating the particles
        self.particles = []
        for i in range(num_particle):
            velocity = new_input[i] - audio
            self.particles.append(Particle(new_input[i], cl[i], velocity))

        self.global_min_cost = min(cl)
        optimal_index = np.where(cl == min(cl))
        print("================================================")
        print(new_input)
        self.gbest_position = new_input[optimal_index][0]
        print("================================================")
        print("global best position: " + str(self.gbest_position))

    def print_positions(self):
        for particle in self.particles:
            particle.__str__()

    def print_best_audio(self):
        print("Current decoded word: " + self.ds.stt(self.gbest_position.astype(np.int16), 16000) + "\t" + "Cost: " + str(self.global_min_cost))

    def update(self, print_toggle):        
        audios = []
        # moving the particles
        for particle in self.particles:
            particle.velocity = self.w * particle.velocity + (self.c1 * np.random.random()) * (particle.pbest_position - particle.position) + (self.c2 * np.random.random()) * (self.gbest_position - particle.position)
            particle.move_particle()
            audios.append(particle.position)

        # calculate new cost
        new_input, cl = self.attack.attack(audios, self.lengths, [[toks.index(x) for x in self.target]] * len(audios), print_toggle)

        # update my particles
        for i, particle in enumerate(self.particles):
            if cl[i] < particle.min_cost:
                particle.min_cost = cl[i]
                particle.pbest_position = new_input[i]

            if cl[i] < self.global_min_cost:
                self.gbest_position = new_input[i]
                self.global_min_cost = cl[i]

        self.w -= 0.005

    #
    def build_model(self, model_path):

        # Build deepspeech model to use for adversarial sample evaluation
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

'''
Particle Class is used to track the individual particle's position,
velocity and personal best position. Instantiated with the same 3 
variables. 
'''
class Particle():
    def __init__(self, position, cost, velocity):
        self.position = position
        self.min_cost = cost
        self.pbest_position = position
        self.velocity = velocity

    '''
    Called by the update function in PSOEnvironment to update the 
    position of each particle.
    '''
    def move_particle(self):
        self.position = self.position + self.velocity

    def __str__(self):
        print("Position: " + str(self.position) + "\nCost: " + str(self.min_cost))


# This is the main function. None of the algorithm takes place here.
# Starts by taking in commandline arguments
# Instantiate the PSOEnvironment class
# Iterate the attack using the PSOEnvironment object
def main():
    mutation_range = 100
    
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--input', type=str, dest="input",
                        required=True,
                        help="Input audio .wav file(s), at 16KHz (separated by spaces)")
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
    model_path = args.model_path
    target = args.target

    with tf.Session() as sess:
        audios = []
        lengths = []

        # Load the inputs that we're given
        fs, audio = wav.read(args.input)
        assert fs == 16000
        assert audio.dtype == np.int16

        # Instantiate the PSOEnvironment object
        pso_environment = PSOEnvironment(population_size, audio, model_path, target, sess)

        for i in range(args.iterations):
            print_toggle = False
            print("Iteration: " + str(i))
            if (i+1) % 10 == 0:
                print_toggle = True  

            # Update the particle position
            pso_environment.update(print_toggle)
            pso_environment.print_best_audio()

        # Save and output the audio file
        wav.write(args.out, 16000, pso_environment.gbest_position.astype(np.int16))
        
main()
