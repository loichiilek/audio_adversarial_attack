## Additional files needed:

DeepSpeech

- git clone https://github.com/mozilla/DeepSpeech.git
- cd DeepSpeech; 
- git checkout tags/v0.4.1



Download the DeepSpeech model

- wget https://github.com/mozilla/DeepSpeech/releases/download/v0.4.1/deepspeech-0.4.1-checkpoint.tar.gz
- tar -xzf deepspeech-0.4.1-checkpoint.tar.gz


## Running the scripts:

Classify audio using the pretrained model

- python3 classify.py --input adversarial_samples/experience.wav


Run the attack

- python3 attack_pso.py --input audio.wav --target deterrence --out adv.wav --iterations 150 --population 200 --model_path deepspeech-0.4.1-checkpoint

- python3 attack_ga.py --input audio.wav --target deterrence --out adv.wav --iterations 150 --population 200 --model_path deepspeech-0.4.1-checkpoint

