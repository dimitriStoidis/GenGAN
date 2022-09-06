
# Generating gender-ambiguous voices for privacy-preserving speech recognition.


<p>Gender-ambiguous speech synthesis is our proposed method for privacy-preserving speech recognition. GenGAN is a generative adversarial network that synthesises mel-spectrograms that are able to convey the content information in speech and conceal gender and identity information.
We provide our pre-trained GenGAN synthesiser and our pre-trained model for gender recognition.</p>

![GenGAN pipeline](/gengan_pipeline.jpg)



# Installation

## Requirements
* Python >= 3.7.4
* PyTorch >= 1.2.0
* hydra-core >= 1.1.1
* Soundfile >= 0.10.3
* librosa >= 0.7.2



## Download data
We use the clean-100 partition of the [LibriSpeech dataset](https://www.openslr.org/12).


## Instructions

0. Clone repository</br>
`git clone https://github.com/dimitriStoidis/GenGAN.git`

1. From a terminal or an Anaconda Prompt, go to project's root directory
and run:</br>
`conda create gengan` </br>
`conda activate gengan` </br>
and install the required packages

2. Download MelGAN neural vocoder and add to `models` directory
`https://github.com/descriptinc/melgan-neurips/tree/master/models`

For training:</br>
3. Create the json manifests to read the data in `/data_files` folder
* speaker and gender labels
* path-to-audio file 

## Training example

To train the model run: </br>
`python train.py --trial model1 --epochs 25 --batch_size 25`


## Demo
To try-out GenGAN on your audio samples:

1. Clone the repository </br>

2. Load the pre-trained GenGAN model from the checkpoint: </br>
 `/models/netG_epoch_25.pt` folder for speech synthesis.

3. Run:</br>
`python demo.py --path_to_audio ./audio/xyz.wav --path_to_models ./models` </br>
The output is a `.wav` file saved in `/audio_` directory.

4. Download and add to path:
the [multi-speaker](https://github.com/descriptinc/melgan-neurips/tree/master/models) pre-trained MelGAN model.


## Evaluation

### Gender Recognition
To perform gender recognition on your saved samples:

1. Load the pre-trained model from the checkpopint: </br>
`/models/model.ckpt-90_GenderNet.pt`

2. Run: </br>
`python GenderNet.py --batch_size bs --set test`</br>



### Speaker Verification
We use the pre-trained SpeakerNet model from here [SpeakerNet](https://github.com/clovaai/voxceleb_trainer) to perform the speaker verification task.

### Automatic Speech Recognition
Download QuartzNet model from: [NeMo](https://catalog.ngc.nvidia.com/orgs/nvidia/models/nemospeechmodels)

### References
The work is based on:
* [PCMelGAN](https://github.com/daverics/pcmelgan)
* [QuartzNet](https://catalog.ngc.nvidia.com/orgs/nvidia/models/nemospeechmodels)
* [MelGAN](https://github.com/descriptinc/melgan-neurips)

### Cite
```
  @misc{https://doi.org/10.48550/arxiv.2207.01052,
  doi = {10.48550/ARXIV.2207.01052},
  url = {https://arxiv.org/abs/2207.01052},
  author = {Stoidis, Dimitrios and Cavallaro, Andrea},
  title = {Generating gender-ambiguous voices for privacy-preserving speech recognition},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution 4.0 International}
}
```
Accepted for publication at Interspeech.

### Contact
For any enquiries contact dimitrios.stoidis@qmul.ac.uk.

### Licence
This work is licensed under the [MIT License](https://github.com/dimitriStoidis/GenGAN/blob/main/LICENSE).
