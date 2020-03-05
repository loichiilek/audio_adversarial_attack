from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import librosa

from pydub import effects
from pydub import AudioSegment
from pydub.generators import WhiteNoise
import soundfile
import random
import time
import sys
import numpy as np
import itertools
import os
import pydub


class Mutators():


    # Volume Change [-5, 10]
    def audio_volume(data, params):
        return data + params

    # threshold [-20,20]
    def audio_comdyrange( data, params):
        return effects.compress_dynamic_range(data, threshold=params)

    # specifies which channel (left or right) to reverse the phase
    def audio_invert_phase( data, params):
        ch = (1, 1)
        if data.channels == 2:
            clis = [(1, 0), (0, 1), (1, 1)]
            ch = clis[random.randint(0, 2)]
        return effects.invert_phase(data, channels=ch)

    # [500, 2000] allow lower pass
    def audio_low_pass_filter( data, params):
        return effects.low_pass_filter(data, cutoff=params)

    # [1, 10000] allow higher pass
    def audio_high_pass_filter( data, params):
        return effects.high_pass_filter(data, cutoff=params)

    # pan [-0.5,1.0]
    def audio_pan( data, params):
        return effects.pan(data, params)

    def audio_append(data, params):
        return data.append(data, crossfade=0)

    def audio_combile( data, data2, params):

        data2 -= params
        return data.overlay(data2)
    def audio_speedup(data, params):
        # Manually override the frame_rate. This tells the computer how many
        # samples to play per second
        sound_with_altered_frame_rate = data._spawn(data.raw_data, overrides={
            "frame_rate": int(data.frame_rate * params)
        })

        # convert the sound with altered frame rate to a standard frame rate
        # so that regular playback programs will work right. They often only
        # know how to play audio at standard frame rate (like 44.1k)
        return sound_with_altered_frame_rate.set_frame_rate(data.frame_rate)
    def audio_speedup1( data, params):
        data_stretch = effects.speedup(data, playback_speed=params)
        return data_stretch

    # # normalize [
    # def audio_normalize( data, params):
    #     return effects.normalize(data, params)

    # Add white noise  [-0.005, +0.005]
    def audio_whitenoise( data, params):
        # print("enter whitenoise")

        wn = np.random.randn(len(data))
        data_wn = data + params * wn
        # print(params)
        return data_wn

    # Extract harmonic elements from an audio time-series. [1.0 , 1.2]

    def audio_harmonic( data, params):
        H, P = librosa.decompose.hpss(data)
        # data_harmonic = librosa.effects.harmonic(data, margin=params)
        return librosa.stft(H)

    # Extract percussive elements from an audio time-series. [1.0, 8.0]

    def audio_percussive( data, params):
        data_per = librosa.effects.percussive(data, margin=params)
        return librosa.istft(data_per)

    # Pitch Shift [-3.0. 8.0] [10, 40]
    def audio_pitch_shift( data, params, ):
        data_pitch_shift = librosa.effects.pitch_shift(data, 16000, params[0], params[1])
        return data_pitch_shift

    # Trim leading and trailing silence from an audio signal. [30, 60]
    def audio_trim( data, params):
        y, index = librosa.effects.trim(data.astype(np.float64), top_db=params)
        return y

    # Change Speed [0.5,1.5]
    # def audio_speedup( data, params):
    #     data_stretch = librosa.effects.time_stretch(data, params)
    #     return data_stretch

    # # Shifting the sound
    # def audio_shift( data, params):
    #     data_roll = np.roll(data, params)
    #     return data_roll

    transformations = [audio_whitenoise, audio_trim,
                       audio_speedup, audio_volume, audio_comdyrange, audio_low_pass_filter, audio_high_pass_filter,
                       audio_append]

    params = []
    params.append(list(map(lambda x: x * 0.001, list(range(-5, 6)))))  # audio_whitenoise
    # params.append(list(map(lambda x: x * 0.01, list(range(100, 105)))))  # audio_harmonic
    # params.append(list(map(lambda x: x * 1.0, list(range(1, 9))))) # audio_percussive


    #params.append(list(itertools.product(list(x * 1.0 for x in range(0, 3)),list(range(10,15)))))  # audio_pitch_shift

    params.append(list(range(20, 80)))  # audio_trim
    params.append(list(map(lambda x: x * 0.01, list(range(80, 130)))))  # audio_speed




    params.append(list(range(-2, 15)))  # audio_volume

    params.append(list(map(lambda x: x * 0.1, list(range(-20, 20)))))  # audio_comdyrange
    # params.append([1])  # audio_invert_phase empty param
    params.append(list(map(lambda x: x * 100, list(range(5, 15)))))  # audio_low_pass_filter
    params.append(list(map(lambda x: x * 500+1, list(range(0, 5)))))  # audio_high_pass_filter

    # params.append(list(map(lambda x: x * 0.1, list(range(-5, 10)))))   # audio_pan
    # params.append(list(range(40, 50)))  # audio_combile

    params.append(list(map(lambda x: 1000+ x * 100, list(range(0, 5)))))  # audio_append

    volume_effect = [audio_volume, audio_low_pass_filter, audio_high_pass_filter]
    speed_effect = [ audio_speedup]
    clear_effect = [audio_whitenoise]

    librosa_transforms = [audio_whitenoise, audio_pitch_shift, audio_trim,audio_harmonic,audio_percussive]

    def audiosegment_to_ndarray(audiosegment):
        samples = audiosegment.get_array_of_samples()
        samples_float = librosa.util.buf_to_float(samples, n_bytes=2,
                                                  dtype=np.float64)
        if audiosegment.channels == 2:
            sample_left = np.copy(samples_float[::2])
            sample_right = np.copy(samples_float[1::2])
            sample_all = np.array([sample_left, sample_right])
        else:
            sample_all = samples_float

        return [sample_all, audiosegment.frame_rate]

    @staticmethod
    def mutate_one(audio_data, audio, cl, which=-1):
        random.seed(time.time())

        length = len(Mutators.transformations)
        strs = list(bin(cl)[2:].zfill(length))
        chosed = [i for i in range(length) if strs[i] == '0']

        if not chosed:
            print('Empty list')
            return  audio_data, 0, cl, False

        if which > 0:
            id = which
        else:
            id = random.choice(chosed)

        strs[id] = '1'
        transformation = Mutators.transformations[id]

        if transformation in Mutators.volume_effect:
            for j in Mutators.volume_effect:
                strs[ Mutators.transformations.index(j)] = '1'
        if transformation in Mutators.speed_effect:
            for j in Mutators.speed_effect:
                strs[Mutators.transformations.index(j)] = '1'
        if transformation in Mutators.clear_effect:
            for j in Mutators.clear_effect:
                strs[Mutators.transformations.index(j)] = '1'
        # print(strs)

        new_cl = int("".join(strs), 2)





        params = Mutators.params[id]
        param = random.sample(params, 1)[0]


        if transformation in Mutators.librosa_transforms:
            data = audio_data
        else:
            data = audio

        new_audio = transformation(data,param)

        if transformation not in Mutators.librosa_transforms:
            new_audio, _ = Mutators.audiosegment_to_ndarray(new_audio)


        return new_audio, strs.count('0'), new_cl, True

    @staticmethod
    def audio_random_mutate(audio, data, sr):
        '''
        This is the interface to perform random mutation on input image, random select
        an mutator and perform a random mutation with a random parameter predefined.

        :param img: input image cl: class
        :param params:
        :return:
        '''

        # test = seed.fname
        # cl = seed.clss

        # audio = AudioSegment.from_wav(test)
        # data, sr = soundfile.read(test)

        # cl what is it change
        cl = int('0', 2)
        batches = []
        cl_batches = []
        space_batch = []

        for i in range(10):
            new_data, space, new_cl, changed = Mutators.mutate_one(data, audio, cl)

            if changed:
                batches.append(new_data)
                cl_batches.append(new_cl)
                space_batch.append(space)

        return (space_batch, np.asarray(batches), cl_batches, sr)




def random_audio_generator(dir, out_dir, num=1000):

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    log_path = os.path.join(out_dir, 'mutation.log')
    log_file = open(log_path, 'w+')
    audios = os.listdir(dir)
    audios.remove('mutation.log')

    for i in range(num):
        cur = random.choice(audios)
        source_path = os.path.join(dir, cur)
        audio = AudioSegment.from_wav(source_path)
        # print(audio.rms)
        data, sr = soundfile.read(source_path)

        new_audio, space, transform, param = Mutators.mutate_one(data, audio, int('0', 2))

        if transform is not None:
            if cur.startswith("id"):
                cur_id = cur.split(",")[0].split(":")[-1]
                out = 'id:new_' + str(i) + ",src:" + cur_id + '.wav'
                # print(cur_id)
            else:
                out = 'id:new_' + str(i) + ",src:" + cur
            audios.append(out)
            soundfile.write(os.path.join(out_dir, out), new_audio, sr)
            log_file.write('{0} -> {1} : {2},  {3}\n'.format(cur, out, transform, param))

    log_file.close()


def random_one_generator(dir, dir_out, which, num=1000):
    if not os.path.exists(dir_out):
        os.makedirs(dir_out)

    log_path = os.path.join(dir_out, 'mutation.log')

    audios = os.listdir(dir)
    log_file = open(log_path, 'w+')
    # cur = audios[0]


    for i in range(num):
        cur = random.choice(audios)
        source_path = os.path.join(dir, cur)
        audio = AudioSegment.from_wav(source_path)
        # print(audio.rms)
        data, sr = soundfile.read(source_path)

        new_audio, space, transform, changed = Mutators.mutate_one(data, audio, int('0', 2), which=which)
        # print(type(data))
        # calculate the singnal to noise rate of the new audio

        if changed:
            # out = 'new_' + str(i) + '.wav'
            if cur.startswith("id"):
                cur_id = cur.split(",")[0].split(":")[-1]
                out = 'id:new_' + str(i) + ",src:" + cur_id + '.wav'
                # print(cur_id)
            else:
                out = 'id:new_' + str(i) + ",src:" + cur

            # audios.append(out)
            soundfile.write(os.path.join(dir_out, out), new_audio, sr)
            log_file.write('{0} -> {1} : {2}\n'.format(cur, out, transform))


    log_file.close()






if __name__ == '__main__':

    random.seed(time.time())

    random_one_generator('seed', "append_samples", 7, num=1)
    # random_audio_generator('RQ1-sample-2', 100)

    # full_trans = Mutators.transformations.copy()

    # for i in range(len(Mutators.transformations)):
    #     # Mutators.transformations = [full_trans[i]]
    #     random_one_generator('RQ1-seeds', "RQ1-samples-"+Mutators.transformations[i].__name__, i, num=12000)
    #     # break

    # p_folder = '/media/lyk/DATA/DeepHunter/DeepSpeech/RQ1-samples'
    # for folder in os.listdir(p_folder):
    #     print(folder)
    #     random_one_generator(os.path.join(p_folder, folder), 100)


    # q = Mutators()
    #
    #
    # data2 = AudioSegment.from_wav('../test_audios/sample.wav')
    # from shutil import copyfile
    #
    # copyfile('../test_audios/sample.wav', '../test_audios/haha.wav')
    #
    # # arr, _ = audiosegment_to_ndarray(data)
    #
    # # list = [audio_whitenoise, audio_harmonic, audio_percussive, audio_pitch_shift, audio_trim, audio_speedup,
    # #                    audio_volume, audio_comdyrange, audio_invert_phase, audio_low_pass_filter, audio_high_pass_filter]
    #
    # arr2, sr = soundfile.read('../test_audios/haha.wav', dtype='int16')
    # # import scipy.io.wavfile as wav
    # # sr3, arr3 = wav.read('../test_audios/haha.wav')
    #
    # # arr2, _ = librosa.effects.trim(arr2, top_db=10)
    # # soundfile.write('../test_audios/haha.wav', arr2, sr)
    #
    # arr2, sr = soundfile.read('../test_audios/haha.wav')
    # soundfile.write('../test_audios/haha.wav', arr2, sr)
    # arr3, sr3 = soundfile.read('../test_audios/haha.wav')
    # exit(0)
    #
    #
    # for i in range(10):
    #     arr2, sr = soundfile.read('../test_audios/haha.wav',dtype='int16')
    #     data = AudioSegment.from_wav('../test_audios/haha.wav')
    #
    #     arr2 = q.random_mutate(arr2,data)
    #     soundfile.write('../test_audios/haha.wav', arr2, sr)
    #
    #
    # # # arr3 = np.concatenate([13.0], arr2)
    # # arr3 = np.insert(arr2, 0, 0.13)
    # # # data2 = ndarray_to_audiosegment(arr2, data.frame_rate)
    # # # play(data2)
    # #
    # # soundfile.write('../test_audios/test.wav', arr3, sr)
    # # arr4, sr4 = soundfile.read('../test_audios/test.wav')
    # # play(data)
    # # q.random_mutate('../test_audios/sample.wav','', 0)
    # exit(0)

    #
    #
    #
    #
    # str = bin(0)[2:].zfill(5)
    # chosed = [i for i in range(5) if str[i] == '0']
    # # print(list(itertools.product(list(x * 1.0 for x in range(-3, 9)),list(range(10,30)))))
    # from pydub.playback import play
    #
    # print("main Test.")
    # m = Mutators()
    # data, sr = soundfile.read('../test_audios/sample.wav')  # librosa.core.load('../test_audios/sample.wav', sr=16000)
    # song = AudioSegment.from_wav('../test_audios/sample.wav')
    # song2 = AudioSegment.from_wav('../test_audios/sample2.wav')
    # # Test speedup
    #
    # # start = time.time()
    # # song = effects.speedup(song, playback_speed=1.5)
    # # print(time.time()-start)
    # # start = time.time()
    # # test_data = m.audio_timestretch(data, 1.5)
    # # print(time.time() - start)
    #
    # # Test noise
    #
    # # start = time.time()
    # # test_data = m.audio_whitenoise(data, 0.005)
    # # print(time.time() - start)
    # #
    # #
    # # start = time.time()
    # # noise = WhiteNoise().to_audio_segment(duration=10000, volume=-30)
    # # song = AudioSegment.from_wav('../test_audios/sample.wav')
    # #
    # # song = song.overlay(noise, loop=True)
    # # print(time.time() - start)
    #
    # # Test volume
    # q = m.audio_harmonic(data, 1.0)
    # test_data = m.audio_harmonic(q,1.0)
    #
    # soundfile.write('../test_audios/pydub.wav', q , sr)
    # # librosa.output.write_wav('../test_audios/librosa.wav',test_data, sr)
    # # song.export('../test_audios/pydub.wav', format='wav')
    # print(sr)
