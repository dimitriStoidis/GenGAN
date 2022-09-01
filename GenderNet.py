from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import json
import random
from pathlib import Path
from tqdm import tqdm
import argparse
from librosa.core import load
import soundfile as sf
from librosa.util import normalize
from hydra import utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", required=False, default="/wavs")
    parser.add_argument("--load_path", default="trial1/run_0/examples/audio/")
    parser.add_argument("--data_path", default="./", type=Path)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seq_len", type=int, default=8192)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--set", type=str, default='train')
    args = parser.parse_args()
    return args


class GenderNet(nn.Module):
    def __init__(self, filters, layers):
        super(GenderNet, self).__init__()
        self.filters = filters
        self.layers = layers
        self.receptive_field = 3 ** layers

        self.initialconv = nn.Conv1d(1, filters, 3, dilation=1, padding=1)
        self.initialbn = nn.BatchNorm1d(filters)

        for i in range(layers):
            setattr(
                self,
                'conv_{}'.format(i),
                nn.Conv1d(filters, filters, 3, dilation=1, padding=1)
            )
            setattr(
                self,
                'bn_{}'.format(i),
                nn.BatchNorm1d(filters)
            )

        self.finalconv = nn.Conv1d(filters, filters, 3, dilation=1, padding=1)

        self.output = nn.Linear(filters, 2)

    def forward(self, x):

        x = self.initialconv(x)
        x = self.initialbn(x)

        for i in range(self.layers):
            x = F.relu(getattr(self, 'conv_{}'.format(i))(x))
            x = getattr(self, 'bn_{}'.format(i))(x)
            x = F.max_pool1d(x, kernel_size=3, stride=3)

        x = F.relu(self.finalconv(x))

        x = F.max_pool1d(x, kernel_size=x.size()[2:])
        x = x.view(-1, self.filters)

        x = torch.sigmoid(self.output(x))

        return x


def files_to_list(filename):
    """
    Takes a text file of filenames and makes a list of filenames
    """
    with open(filename, encoding="utf-8") as f:
        files = f.readlines()

    files = [f.rstrip() for f in files]
    return files


class AudioDataset(torch.utils.data.Dataset):
    """
    This is the main class that calculates the spectrogram and returns the
    spectrogram, audio pair.
    """

    def __init__(self, training_files, segment_length, sampling_rate, augment=True):
        self.sampling_rate = sampling_rate
        self.segment_length = segment_length
        self.audio_files = files_to_list(training_files)
        self.audio_files = [Path(training_files).parent / x for x in self.audio_files]
        random.shuffle(self.audio_files)
        self.augment = augment
        with open("libri_train_speakers.json") as file:
            self.speakers = sorted(json.load(file))
        with open("libri_train_gender.json") as file:
            self.gender = json.load(file)

    def __getitem__(self, index):
        # Read audio
        filename = self.audio_files[index]
        speaker = str(filename).split("/")[-1].split("-")[0]
        speaker_ind = self.speakers.index(str(speaker))
        audio, sampling_rate = self.load_wav_to_torch(str(filename))
        # Take segment
        if audio.size(0) >= self.segment_length:
            max_audio_start = audio.size(0) - self.segment_length
            audio_start = random.randint(0, max_audio_start)
            audio = audio[audio_start : audio_start + self.segment_length]
        else:
            audio = F.pad(
                audio, (0, self.segment_length - audio.size(0)), "constant"
            ).data
        print(audio.shape)
        return audio.unsqueeze(0), speaker, self.gender[speaker_ind]

    def __len__(self):
        return len(self.audio_files)

    def load_wav_to_torch(self, full_path):
        """
        Loads wavdata into torch array
        """
        data, sampling_rate = load(full_path, sr=self.sampling_rate)
        data = 0.95 * normalize(data)

        if self.augment:
            amplitude = np.random.uniform(low=0.3, high=1.0)
            data = data * amplitude

        return torch.from_numpy(data).float(), sampling_rate


class AudioDataset2(torch.utils.data.Dataset):
    """
    This is the main class that calculates the spectrogram and returns the
    spectrogram, audio pair.
    """

    def __init__(self, training_files, segment_length, sampling_rate, augment=True):
        self.sampling_rate = sampling_rate
        self.segment_length = segment_length
        self.audio_files = files_to_list(training_files)
        self.audio_files = [Path(training_files).parent / x for x in self.audio_files]
        random.shuffle(self.audio_files)
        self.augment = augment
        with open("libri_test_speakers.json") as file:
            self.speakers = sorted(json.load(file))
        with open("libri_test_gender.json") as file:
            self.gender = json.load(file)

    def __getitem__(self, index):
        # Read audio
        filename = self.audio_files[index]
        speaker = str(filename).split("/")[-1].split("-")[0]
        speaker_ind = self.speakers.index(str(speaker))
        audio, sampling_rate = self.load_wav_to_torch(str(filename))
        # Take segment
        if audio.size(0) >= self.segment_length:
            max_audio_start = audio.size(0) - self.segment_length
            audio_start = random.randint(0, max_audio_start)
            audio = audio[audio_start: audio_start + self.segment_length]
        else:
            audio = F.pad(
                audio, (0, self.segment_length - audio.size(0)), "constant"
            ).data

        return audio.unsqueeze(0), speaker, self.gender[speaker_ind]

    def __len__(self):
        return len(self.audio_files)

    def load_wav_to_torch(self, full_path):
        """
        Loads wavdata into torch array
        """
        data, sampling_rate = load(full_path, sr=self.sampling_rate)
        data = 0.95 * normalize(data)

        if self.augment:
            amplitude = np.random.uniform(low=0.3, high=1.0)
            data = data * amplitude

        return torch.from_numpy(data).float(), sampling_rate


def save_checkpoint(classifier, optimizer, epoch, checkpoint_dir):
    checkpoint_state = {
        "classifier": classifier.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch
    }
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    checkpoint_path = checkpoint_dir / "model.ckpt-{}_GenderNet_Net.pt".format(epoch)
    torch.save(checkpoint_state, checkpoint_path)
    print("Saved checkpoint: {}".format(checkpoint_path.stem))

    return checkpoint_path


# Create the training function
def train(train_dataloader, model, loss, optimizer):
	model.train()
	args.set = 'train'
	size = len(train_dataloader.dataset)
	for i, (mels, speaker, gen) in enumerate(tqdm(train_dataloader), 1):

		optimizer.zero_grad()
		pred = model(mels.to(device))
		loss = cost(pred, gen.to(device))
		loss.backward()
		optimizer.step()
		loss, current = loss.item(), i * len(mels)
		print('CE loss: {:.5f}  [{:.2f}%]'.format(loss, (current/size)*100))


def test(dataloader, model, checkpoint_path):
	print("Resume checkpoint from: {}:".format(checkpoint_path))
	resume_path = utils.to_absolute_path(checkpoint_path)
	checkpoint = torch.load(resume_path, map_location=lambda storage, loc: storage)
	model.load_state_dict(checkpoint["classifier"])
	print(model.load_state_dict(checkpoint["classifier"]))
	optimizer.load_state_dict(checkpoint["optimizer"])
	start_epoch = checkpoint["epoch"]
	print("\nResuming from epoch: ", start_epoch)

	size = len(dataloader.dataset)
	model.eval()
	test_loss, correct = 0, 0

	with torch.no_grad():
		for i, (mels, speaker, gen) in enumerate(tqdm(dataloader), 1):
			pred = model(mels.to(device))
			test_loss += cost(pred, gen.to(device)).item()
			correct += (pred.argmax(1) == gen.to(device)).type(torch.float).sum().item()

	test_loss /= size
	correct /= size

	print('\nTest Error:\nacc: {:.5f}, avg loss: {:.5f}\n'.format(100*correct, test_loss))


if __name__ == "__main__":
	args = parse_args()
	seed_val = 123
	torch.manual_seed(seed_val)
	torch.cuda.manual_seed(seed_val)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	random.seed(seed_val)
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	print('Using {} device'.format(device))

	#######################
	# Create data loaders #
	#######################
	manifest = "files_path.txt"
	# train_set = AudioDataset(
	# 	Path(args.data_path) / "train.txt", args.seq_len, sampling_rate=16000
	# )
	test_set = AudioDataset2(
		Path(args.data_path) / manifest,
		args.seq_len,
		sampling_rate=16000,
		augment=True,
	)

	# train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=1, drop_last=True)
	test_loader = DataLoader(test_set, batch_size=128)

	model = GenderNet(16, 5).to(device)
	print(model)

	# loss function
	cost = torch.nn.CrossEntropyLoss()
	learning_rate = 1e-3
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

	epochs = args.epochs

	for t in range(epochs):
		# train(train_loader, model, cost, optimizer)
		print("\n......Testing model....\n..........\n")
		test(test_loader, model, "models/model.ckpt-90_GenderNet_Net.pt")

		if t % 3 == 0:
			checkpoint_path = save_checkpoint(model, optimizer, t, Path('./models'))
