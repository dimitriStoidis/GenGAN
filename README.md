# GenGAN
Generating gender-ambiguous voices for privacy-preserving speech recognition.


![Alt text](/gengan_pipeline.jpg)

=================================================================================

# Installation

## Requirements
* Python = 3.7.4
* PyTorch = 1.2.0
* Hydra
* Soundfile = 0.10.3
* torchaudio
* librosa = 0.7.2
* scipy = 1.3.1
* tqdm


## Download data
We train and test the model on the [[LibriSpeech dataset](https://www.openslr.org/12)] (train-clean-100 and test-clean-100).


## Instructions

0. Clone repository
1. Create manifests to read the data
2. run train and test scripts



## Running example

To train the model run `python train.py --experiment_name trial1`

## Evaluation


### Gender Recognition
Run `python GenderNet.py`
<p>load the pre-trained model </p>

### Speaker Verification
We use the pre-trained SpeakerNet model from here [SpeakerNet](https://github.com/clovaai/voxceleb_trainer) to perform the speaker verification task.

### ASR
Download QuartzNet model from: [NeMo](https://catalog.ngc.nvidia.com/orgs/nvidia/models/nemospeechmodels)

The work is based on PCMelGAN []
[PCMelGAN](https://github.com/daverics/pcmelgan)

### Licence
This work is licensed under the MIT License.
