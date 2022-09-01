# GenGAN

## Generating gender-ambiguous voices for privacy-preserving speech recognition.


<p>Gender-ambiguous generation is our proposed method for privacy-preserving speech recognition. GenGAN is a generative adversarial network that synthesises speech mel-spectrograms that are able to convey the content information in speech and conceal gender and identity information.
We provide our pre-trained GenGAN synthesiser and our pre-trained model for gender recognition.</p>

![GenGAN pipeline](/gengan_pipeline.jpg)

=================================================================================

# Installation

## Requirements
* Python = 3.7.4
* PyTorch = 1.2.0
* hydra-core = 1.1.1
* Soundfile = 0.10.3
* torchaudio = 0.10.0
* librosa = 0.7.2



## Download data
We use the clean-100 partition of the [LibriSpeech dataset](https://www.openslr.org/12).


## Instructions

0. Clone repository
`git clone https://github.com/dimitriStoidis/GenGAN.git`

1. Create the json manifests to read the data in `/data_files` folder
* speaker and gender labels
* path-to-audio file 


## Running example

To train the model run `python train.py --experiment_name trial1`

## Evaluation

Load the pre-trained GenGAN model in `/models` folder for speech synthesis.

### Gender Recognition
Run `python GenderNet.py`
<p>Load the pre-trained model in `/models/model.ckpt-90_GenderNet.pt` for evaluation or train GenderNet from scratch.


### Speaker Verification
We use the pre-trained SpeakerNet model from here [SpeakerNet](https://github.com/clovaai/voxceleb_trainer) to perform the speaker verification task.

### ASR
Download QuartzNet model from: [NeMo](https://catalog.ngc.nvidia.com/orgs/nvidia/models/nemospeechmodels)

The work is based on:
* [PCMelGAN](https://github.com/daverics/pcmelgan)
* [QuartzNet](https://catalog.ngc.nvidia.com/orgs/nvidia/models/nemospeechmodels)
* [MelGAN](https://github.com/descriptinc/melgan-neurips)
### Licence
This work is licensed under the MIT License.
