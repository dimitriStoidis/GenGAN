# GenGAN
Generating gender-ambiguous voices for privacy-preserving speech recognition.

=================================================================================

We train and test the model on the [[LibriSpeech dataset](https://www.openslr.org/12)] (train-clean-100 and test-clean-100).

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



## Instructions
To train the model run <code> `python train.py --experiment_name`</code>
<p>Create manifests to read the data.</p>
0. Clone repository


## Running example


## Evaluation


### Gender Recognition
Run <code>`python GenderNet.py`</code>
<p>load the pre-trained model </p>

### Speaker Verification
We use the pre-trained SpeakerNet model from here [SpeakerNet](https://github.com/clovaai/voxceleb_trainer) to perform the speaker verification task.

### ASR
Download QuartzNet model from: [NeMo](https://catalog.ngc.nvidia.com/orgs/nvidia/models/nemospeechmodels)



### Licence
This work is licensed under the MIT License.
