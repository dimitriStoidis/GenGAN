from tqdm import tqdm
import dataset
import librosa
from torch.utils.data import DataLoader, random_split
import torch
import torch.nn.functional as F
from utils import *
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
import argparse
import yaml
from pathlib import Path
import time
from networks import UNet_GenGAN, AlexNet_Discriminator
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
    parser.add_argument("--trial", type = str, required = True)
    parser.add_argument("--epochs", type = int, default=25)
    parser.add_argument("--batch_size", type = int, default=10)
    parser.add_argument("--save_interval", type = int, default=2)
    parser.add_argument("--checkpoint_interval", type = int, default=5)
    parser.add_argument("--load_path", type = str, default = '/logs/')
    parser.add_argument("--resume_trial", type = bool, default = False)
    parser.add_argument("--G_lr", type = float, default = 3e-4)
    parser.add_argument("--D_lr", type = float, default = 3e-4)
    parser.add_argument("--utility_loss", type = bool, default = False)

    # Model and loss parameters
    parser.add_argument("--loss", type = str, default = None)
    parser.add_argument("--eps", type = float, default=10)
    parser.add_argument("--filter_receptive_field", type = int, default = 3)
    parser.add_argument("--n_mel_channels", type = int, default = 80)
    parser.add_argument("--ngf", type = int, default = 32)
    parser.add_argument("--n_residual_layers", type = int, default=3)
    parser.add_argument("--sampling_rate", type = int, default=16000)
    parser.add_argument("--seeds", type = int, nargs = '+', default =123)
    parser.add_argument("--num_runs", type = int, default = 1)
    parser.add_argument("--n_completed_runs", type = int, default = 1)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    root = Path(os.getcwd())
    experiment_name = args.trial
    print('******Experiment NAME*******\n', experiment_name)

    num_runs = args.num_runs
    device = 'cuda:' + str(args.device)
    log_dir = os.path.join(root, 'logs')
    experiment_dir = os.path.join(log_dir, experiment_name)

    print("Resume Experiment?", args.resume_trial)
    if os.path.exists(experiment_dir):
        print("Experiment with this name already exists, use --resume_trial to continue.")
    else:
        os.mkdir(experiment_dir)

    # hyper parameters
    num_genders = 2
    eps = args.eps
    batch_train = 24
    noise_dim = 65
    manualSeed = 1038
    set_seed(manualSeed)

    dataset1 = dataset.LibriDataset(
        root='./',
        hop_length=160,
        sr=16000,
        set='train',
        sample_frames=32)

    train_loader = DataLoader(
        dataset1,
        batch_size=batch_train,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True)

    dataset2 = dataset.LibriDataset(
        root='./',
        hop_length=160,
        sr=16000,
        set='test',
        sample_frames=32)

    save_loader = DataLoader(
        dataset2,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False)

    n_train = dataset1.__len__()

    # Load MelGAN vocoder
    fft = Audio2Mel(sampling_rate = args.sampling_rate)
    Mel2Audio = MelGAN_Generator(args.n_mel_channels, args.ngf, args.n_residual_layers).to(device)
    Mel2Audio.load_state_dict(torch.load('models/multi_speaker.pt'))

    # Loss functions
    distortion_loss = nn.MSELoss()
    adversarial_loss = nn.CrossEntropyLoss()
    adversarial_loss_rf = nn.CrossEntropyLoss()

    for run in range(num_runs):
        run_dir = os.path.join(experiment_dir, 'run_' + str(run))
        checkpoint_dir = os.path.join(run_dir, 'checkpoints')
        example_dir = os.path.join(run_dir, 'examples')
        example_audio_dir = os.path.join(example_dir, 'audio')

        if not args.resume_trial:
            os.mkdir(run_dir)
            os.mkdir(example_dir)
            os.mkdir(checkpoint_dir)
            os.mkdir(example_audio_dir)

        # Set random seed
        set_seed(args.seeds)

        with open(Path(run_dir) / "args.yml", "w") as f:
            yaml.dump(args, f)
            yaml.dump({'Seed used' : manualSeed}, f)
            yaml.dump({'Run number' : run}, f)

        # Load trainable models GenGAN generator and discriminator
        netG = UNet_(1, 1, chs=[8, 16, 32, 64, 128],
                          kernel_size=args.filter_receptive_field,
                          image_width=32, image_height=80, noise_dim=noise_dim,
                          nb_classes=2, embedding_dim=16, use_cond = False).to(device)
        netD = AlexNet_Discriminator(num_genders + 1).to(device)

        # Optimizers
        optG = torch.optim.Adam(netG.parameters(), args.G_lr, betas = (0.5, 0.99))
        optD = torch.optim.Adam(netD.parameters(), args.D_lr, betas = (0.5, 0.99))

        # Put training objects into list for loading and saving state dicts
        training_objects = []
        training_objects.append(('netG',  netG))
        training_objects.append(('optG',  optG))
        training_objects.append(('netD', netD))
        training_objects.append(('optD', optD))
        training_objects.sort(key = lambda x : x[0])

        # Load from checkpoints
        start_epoch = 0
        if args.resume_trial:
                start_epoch = 100
                print('Resuming experiment {} from checkpoint, {} epochs completed.'.format(args.trial, start_epoch))
                netG.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'netG_latest_epoch_20.pt')))
                netD.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'netD_latest_epoch_20.pt')))

        print("GAN training initiated, {} epochs".format(args.epochs))
        script_start = time.time()
        for epoch in range(start_epoch, args.epochs + start_epoch):

            # Add variables to add batch losses to
            G_distortion_loss_accum = 0
            G_adversary_loss_accum = 0
            D_real_loss_accum = 0
            D_fake_loss_accum = 0

            epoch_start = time.time()
            netG.train()
            netD.train()
            for i, (x, _, gender, utterance) in tqdm(enumerate(train_loader)):

                gender = gender.to(device)
                x = torch.unsqueeze(x, 1)
                spectrograms = fft(x).detach()
                spectrograms, means, stds = preprocess_spectrograms(spectrograms)
                spectrograms = torch.unsqueeze(spectrograms, 1).to(device)

                # ------------------------
                # Train Generator (Real/Fake)
                # ------------------------
                optG.zero_grad()

                z2 = torch.randn(spectrograms.shape[0], noise_dim*5).to(device)

                # randomly sample from synthetic gender distribution
                gen_secret = Variable(LongTensor(np.random.choice([1.0], spectrograms.shape[0]))).to(device)
                gen_secret = gen_secret * np.random.normal(0.5, math.sqrt(0.05))
                # pass M' + Z2 + Y_N gender to Generator and get M'
                # -----------------------------------------------------------------
                # spectrogram directly in G (no filtering step)
                gen_mel = netG(spectrograms, z2, gen_secret)

                # Gender prediction from M''
                pred_secret = netD(gen_mel)
                # Utility MSE generated spectrogram and GT spec
                generator_distortion_loss = distortion_loss(gen_mel, spectrograms)

                G_distortion_loss_accum += generator_distortion_loss.item()
                # La loss predicted gender close to 0.5
                generator_adversary_loss = adversarial_loss(pred_secret, gender.long().to(device))  

                G_adversary_loss_accum += generator_adversary_loss.item()

                # ----------- Generator loss--------------
                netG_loss = generator_distortion_loss + generator_adversary_loss * eps

                netG_loss.backward()
                optG.step()

                #  Train Discriminator D --> (Y_{R/F})

                optD.zero_grad()
                # feed GT spectrograms M to generator
                real_pred_secret = netD(spectrograms)

                fake_pred_secret = pred_secret.detach()

                D_real_loss = adversarial_loss_rf(real_pred_secret, gender.long().to(device)).to(device)
                D_genderless_loss = adversarial_loss_rf(fake_pred_secret, gen_secret).to(device)

                D_real_loss_accum += D_real_loss.item()
                D_fake_loss_accum += D_genderless_loss.item()

                netD_loss = D_real_loss + D_genderless_loss
                netD_loss.backward()
                optD.step()

                # ----------------------------------------------
                #   Compute accuracies
                # ----------------------------------------------
                # D accuracy on original gender in real and generated (fake) data,
                # and sampled gender in generated (fake) data

                if i % 100 == 0:
                    print(" \n G {:5.5f} L_d: {:5.5f} Dfake_Yn {:5.5f} \n "
                          "D {:5.5f} D real {:5.5f} D fake {:5.5f}\n --[{:5.2f}]%".
                          format(netG_loss, generator_distortion_loss, generator_adversary_loss, netD_loss, D_real_loss, D_genderless_loss, (i*batch_train*100)/n_train))

            print('\n__________________________________________________________________________')
            print("Epoch {} completed | Time: {:5.2f} s "
                  .format(epoch+1, time.time() - epoch_start))

            # ----------------------------------------------
            #   Save test samples
            # ----------------------------------------------

            if (epoch + 1) % args.save_interval == 0:
                print('\n__________________________________________________________________________')
                print("Saving audio and spectrogram samples.....")
                netG.eval()
                for i, (x, _, gender, utterance) in tqdm(enumerate(save_loader)):
                    x = torch.unsqueeze(x, 1)
                    spectrograms = fft(x).detach()
                    spectrograms, means, stds = preprocess_spectrograms(spectrograms)
                    spectrograms = torch.unsqueeze(spectrograms, 1).to(device)

                    z2 = torch.randn(spectrograms.shape[0], noise_dim*5).to(device)
                    gen_secret = Variable(LongTensor(np.random.choice([1.0], spectrograms.shape[0]))).to(device)
                    neutral = gen_secret * np.random.normal(0.5, math.sqrt(0.05))

                    generated_neutral = netG(spectrograms, z2, neutral).detach()

                    # Predict gender
                    generated_neutral = torch.squeeze(generated_neutral, 1).to(device) * 3 * stds.to(device) + means.to(device)
                    inverted_neutral = Mel2Audio(generated_neutral).squeeze().detach().cpu()
                    f_name_neutral_audio = os.path.join(example_audio_dir, utterance[0] + '.wav'.
                                                        format(epoch + 1))
                    save_sample(f_name_neutral_audio, 16000, inverted_neutral)

                print("Success!")
                print('__________________________________________________________________________\n')

            if (epoch + 1) % args.checkpoint_interval == 0:
                save_epoch = epoch + 1
                old_checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, '*latest*')))
                if old_checkpoints:
                    for i, _ in enumerate(old_checkpoints):
                        os.remove(old_checkpoints[i])
                for name, object in training_objects:
                    torch.save(object.state_dict(), os.path.join(checkpoint_dir, name + '_epoch_{}.pt'.format(save_epoch)))
                    torch.save(object.state_dict(), os.path.join(checkpoint_dir, name + '_latest_epoch_{}.pt'.format(save_epoch)))
            print("Total time: ", script_start)

        print("Run number {} completed.".format(run+1))
        print('__________________________________________________________________________')


if __name__ == "__main__":
    main()
