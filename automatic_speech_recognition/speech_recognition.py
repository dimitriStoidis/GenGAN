import soundfile as sf
import nemo.collections.asr as nemo_asr
import os

# See documentation
# https://catalog.ngc.nvidia.com/orgs/nvidia/models/nemospeechmodels
# Speech-to-Text


def speechtotext(transcripts, folder_path):
    quartznet = nemo_asr.models.ASRModel.restore_from('./models/QuartzNet15x5Base-En.nemo')
    path = [folder_path + f for f in os.listdir(folder_path) if f.endswith(".wav")]
    f = open(transcripts, "w")
    for fname, trans in zip(path, quartznet.transcribe(paths2audio_files=path)):
        print(str(fname).split("audio/")[-1].split(".wav")[0], str(trans))
        f.write(str(fname).split("audio/")[-1].split(".wav")[0] + " , " + str(trans) + "\n")
    f.close()


if __name__ == '__main__':
    speechtotext("filename.txt", "/logs/model1/run_0/examples/audio/")
