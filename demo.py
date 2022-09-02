from tqdm import tqdm
import librosa
from librosa.core import load
from librosa.util import normalize
import torch
import torch.nn.functional as F
from utils import *
import argparse
from pathlib import Path
import time
from networks import UNetFilter
from torch.autograd import Variable
import glob
from modules import MelGAN_Generator, Audio2Mel
from pathlib import Path
import random
import pdb
import math


LongTensor = torch.cuda.LongTensor
FloatTensor = torch.cuda.FloatTensor


def parse_args():
    parser = argparse.ArgumentParser()

    # Training parameters
    parser.add_argument("--device", type = str, default = '0')
    parser.add_argument("--filter_receptive_field", type = int, default = 3)
    parser.add_argument("--n_mel_channels", type = int, default = 80)
    parser.add_argument("--ngf", type = int, default = 32)
    parser.add_argument("--n_residual_layers", type = int, default=3)
    parser.add_argument("--sampling_rate", type = int, default=16000)
    parser.add_argument("--seeds", type = int, nargs = '+', default =123)
    parser.add_argument("--num_runs", type = int, default = 1)
    parser.add_argument("--noise_dim", type=int, default=65)
    parser.add_argument("--max_duration", type=int, default=16.7)
    args = parser.parse_args()
    return args


def load_wav_to_torch(full_path, max_duration):
   audio, sampling_rate = load(full_path, sr=16000)
   audio = 0.95 * normalize(audio)
   duration = len(audio)
   audio = torch.from_numpy(audio).float()
   # utterances of segment_length
   if audio.size(0) <= max_duration*sampling_rate:
       audio = F.pad(audio, (0, int(max_duration*sampling_rate) - audio.size(0)), "constant").data
   return audio, duration


def main():
    args = parse_args()
    root = Path(os.getcwd())
    device = 'cuda:' + str(args.device)

    # hyper parameters
    noise_dim = args.noise_dim
    manualSeed = 1038
    max_duration = args.max_duration
    set_seed(manualSeed)
    audio_path = '/*.wav'
    audio_file = audio_path.split('.wav')[0].split('/')[-1]

    # Load MelGAN vocoder
    fft = Audio2Mel(sampling_rate=args.sampling_rate)
    Mel2Audio = MelGAN_Generator(args.n_mel_channels, args.ngf, args.n_residual_layers).to(device)
    Mel2Audio.load_state_dict(torch.load('/path-to/models/multi_speaker.pt'))

    run_dir = os.path.join(root, 'audio_')
    if not os.path.exists(run_dir):
        os.mkdir(run_dir)

    # Set random seed
    set_seed(args.seeds)

    netG = UNetFilter(1, 1, chs=[8, 16, 32, 64, 128],
              kernel_size=args.filter_receptive_field,
              image_width=32, image_height=80, noise_dim=65,
              nb_classes=2, embedding_dim=16, use_cond=False).to(device)

    # Put training objects into list for loading and saving state dicts
    training_objects = []
    training_objects.append(('netG',  netG))
    training_objects.sort(key=lambda x: x[0])

    # Load from checkpoint
    netG.load_state_dict(torch.load('/path-to/models/netG_epoch_25.pt'))

    print("GenGAN synthesis initiated")
    netG.eval()
    x, dur = load_wav_to_torch(audio_path, max_duration)
    x = torch.unsqueeze(x, 1)
    spectrograms = fft(x.reshape(1, x.size(0))).detach()
    spectrograms, means, stds = preprocess_spectrograms(spectrograms)
    spectrograms = torch.unsqueeze(spectrograms, 1).to(device)

    z = torch.randn(spectrograms.shape[0], noise_dim * 5).to(device)
    gen_secret = Variable(LongTensor(np.random.choice([1.0], spectrograms.shape[0]))).to(device)
    y_n = gen_secret * np.random.normal(0.5, math.sqrt(0.05))
    # Input to Generator
    generated_neutral = netG(spectrograms, z, y_n).detach()

    # spectrogram inversion
    generated_neutral = torch.squeeze(generated_neutral, 1).to(device) * 3 * stds.to(device) + means.to(device)
    inverted_neutral = Mel2Audio(generated_neutral).squeeze().detach().cpu()
    print("Saving audio..")
    f_name_neutral_audio = os.path.join(run_dir, audio_file + '_transformed.wav')
    save_sample(f_name_neutral_audio, args.sampling_rate, inverted_neutral[:dur])


if __name__ == "__main__":
    main()
