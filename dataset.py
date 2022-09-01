import pandas as pd
import json
import librosa
import numpy as np
import torch
import torch.utils.data
import torch.nn.functional as F
import torch.nn as nn
from utils import *
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pathlib import Path
from librosa.core import load
from librosa.util import normalize
import random
import pdb


class LibriDataset(Dataset):
    def __init__(self, root, hop_length, sr, set, sample_frames):
        self.root = Path(root)
        self.hop_length = hop_length
        self.sample_frames = sample_frames
        self.set = set
        self.segment_length = int(16.7*sr)

        with open(self.root / Path("libri_" + str(set) + "_speakers.json")) as file:
            self.speakers = sorted(json.load(file))
        with open(self.root / Path("libri_" + str(set) + "_gender.json")) as file:
            self.gender = json.load(file)
        min_duration = (sample_frames + 2) * hop_length / sr
        with open(self.root / Path("libri_" + str(set) + "_preprocess.json")) as file:
            metadata = json.load(file)
            self.metadata = [
                Path(in_path) for in_path, _, duration in metadata
                if duration > min_duration
            ]

    def __len__(self):
        return len(self.metadata)

    def load_wav_to_torch(self, full_path):
        data, sampling_rate = load(full_path, sr=16000)
        data = 0.95 * normalize(data)
        return torch.from_numpy(data).float(), sampling_rate

    def __getitem__(self, index):
        path = self.metadata[index]
        utterance = str(path).split("wav/")[-1].split(".wav")[0]
        path = self.root.parent / path

        audio, sampling_rate = self.load_wav_to_torch(path)

        # utterances of segment_length
        if audio.size(0) <= self.segment_length:
            audio = F.pad(
                audio, (0, self.segment_length - audio.size(0)), "constant"
            ).data

        speaker = self.speakers.index(path.parts[-1].split("-")[0])
        gender = self.gender[speaker]  # speaker info oly used to get the gender
        gender = torch.tensor(gender)
        return audio, speaker, gender, utterance