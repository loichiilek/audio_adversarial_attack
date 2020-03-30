import argparse
from deepspeech import Model
import soundfile
import sys


parser = argparse.ArgumentParser(description=None)
parser.add_argument('--input', type=str, dest="input",
                    required=True,
                    help="Input audio .wav file at 16KHz")
args = parser.parse_args()
while len(sys.argv) > 1:
    sys.argv.pop()


BEAM_WIDTH = 500
LM_ALPHA = 0.75
LM_BETA = 1.85
N_FEATURES = 26
N_CONTEXT = 9
MODEL_PATH = 'deepspeech-0.4.1-checkpoint/models/output_graph.pb'
ALPHABET_PATH = 'deepspeech-0.4.1-checkpoint/models/alphabet.txt'
LM_PATH = 'deepspeech-0.4.1-checkpoint/models/lm.binary'
TRIE_PATH = 'deepspeech-0.4.1-checkpoint/models/trie'

ds = Model(MODEL_PATH, N_FEATURES, N_CONTEXT, ALPHABET_PATH, BEAM_WIDTH)
ds.enableDecoderWithLM(ALPHABET_PATH, LM_PATH, TRIE_PATH, LM_ALPHA, LM_BETA)

# Audio Path
AUDIO_PATH = args.input
# Read soundfile should say experience
audio_data, sample_rate = soundfile.read(AUDIO_PATH ,dtype='int16')

print(audio_data)
print(ds.stt(audio_data, sample_rate))