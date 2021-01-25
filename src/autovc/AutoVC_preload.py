'''
Created on Jan 4, 2021

@author: micou
'''
import os
import numpy as np
import pickle
import torch
from math import ceil
from src.autovc.retrain_version.model_vc_37_1 import Generator
from pydub import AudioSegment
import pynormalize.pynormalize
from scipy.io import  wavfile as wav
from scipy.signal import stft


from src.autovc.retrain_version.vocoder_spec.extract_f0_func import extract_f0_func_audiofile
from src.autovc.utils import quantize_f0_interp
from thirdparty.resemblyer_util.speaker_emb import get_spk_emb


def match_target_amplitude(sound, target_dBFS):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)

class AutoVC_mel_Convertor():

    def __init__(self, src_dir, autovc_model_path='examples/ckpt/ckpt_autovc.pth', proportion=(0., 1.), seed=0):
        self.src_dir = src_dir
        if(not os.path.exists(os.path.join(src_dir, 'filename_index.txt'))):
            self.filenames = []
        else:
            with open(os.path.join(src_dir, 'filename_index.txt'), 'r') as f:
                lines = f.readlines()
                self.filenames = [(int(line.split(' ')[0]), line.split(' ')[1][:-1]) for line in lines]

        np.random.seed(seed)
        rand_perm = np.random.permutation(len(self.filenames))
        proportion_idx = (int(proportion[0] * len(rand_perm)), int(proportion[1] * len(rand_perm)))
        selected_index = rand_perm[proportion_idx[0] : proportion_idx[1]]
        self.selected_filenames = [self.filenames[i] for i in selected_index]
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.G = Generator(16, 256, 512, 16).eval().to(self.device)

        g_checkpoint = torch.load(autovc_model_path, map_location=self.device)
        self.G.load_state_dict(g_checkpoint['model'])

        print('{} out of {} are in this portion'.format(len(self.selected_filenames), len(self.filenames)))
        
    def __convert_single_only_au_AutoVC_format_to_dataset__(self, filename, build_train_dataset=True):
        """
        Convert a single file (only audio in AutoVC embedding format) to numpy arrays
        :param filename:
        :param is_map_to_std_face:
        :return:
        """

        global_clip_index, video_name = filename

        # audio_file = os.path.join(self.src_dir, 'raw_wav', '{}.wav'.
        #                           format(video_name[:-4]))
        audio_file = os.path.join(self.src_dir, 'raw_wav', '{:05d}_{}_audio.wav'.
                                  format(global_clip_index, video_name[:-4]))
        if(not build_train_dataset):
            import shutil
            audio_file = os.path.join(self.src_dir, 'raw_wav', '{:05d}_{}_audio.wav'.
                                      format(global_clip_index, video_name[:-4]))
            shutil.copy(os.path.join(self.src_dir, 'test_wav_files', video_name), audio_file)

        sound = AudioSegment.from_file(audio_file, "wav")
        normalized_sound = match_target_amplitude(sound, -20.0)
        normalized_sound.export(audio_file, format='wav')

        S, f0_norm = extract_f0_func_audiofile(audio_file, 'M')
        f0_onehot = quantize_f0_interp(f0_norm)
        mean_emb, _ = get_spk_emb(audio_file)

        return S, mean_emb, f0_onehot
    
    def convert_single_wav_to_autovc_input(self, audio_filename):


        def pad_seq(x, base=32):
            len_out = int(base * ceil(float(x.shape[0]) / base))
            len_pad = len_out - x.shape[0]
            assert len_pad >= 0
            return np.pad(x, ((0, len_pad), (0, 0)), 'constant'), len_pad

        
        emb = np.loadtxt('src/autovc/retrain_version/obama_emb.txt')
        emb_trg = torch.from_numpy(emb[np.newaxis, :].astype('float32')).to(self.device)

        aus = []
        audio_file = audio_filename

        sound = AudioSegment.from_file(audio_file, "wav")
        normalized_sound = match_target_amplitude(sound, -20.0)
        normalized_sound.export(audio_file, format='wav')

        
        x_real_src, f0_norm = extract_f0_func_audiofile(audio_file, 'F')
        f0_org_src = quantize_f0_interp(f0_norm)
        emb, _ = get_spk_emb(audio_file)

        ''' normal length version '''
        # x_real, len_pad = pad_seq(x_real_src.astype('float32'))
        # f0_org, _ = pad_seq(f0_org_src.astype('float32'))
        # x_real = torch.from_numpy(x_real[np.newaxis, :].astype('float32')).to(device)
        # emb_org = torch.from_numpy(emb[np.newaxis, :].astype('float32')).to(device)
        # f0_org = torch.from_numpy(f0_org[np.newaxis, :].astype('float32')).to(device)
        # print('source shape:', x_real.shape, emb_org.shape, emb_trg.shape, f0_org.shape)
        #
        # with torch.no_grad():
        #     x_identic, x_identic_psnt, code_real = self.G(x_real, emb_org, f0_org, emb_trg, f0_org)
        # print('converted shape:', x_identic_psnt.shape, code_real.shape)

        ''' long split version '''
        l = x_real_src.shape[0]
        x_identic_psnt = []
        step = 4096
        for i in range(0, l, step):
            x_real = x_real_src[i:i + step]
            f0_org = f0_org_src[i:i + step]

            x_real, len_pad = pad_seq(x_real.astype('float32'))
            f0_org, _ = pad_seq(f0_org.astype('float32'))
            x_real = torch.from_numpy(x_real[np.newaxis, :].astype('float32')).to(self.device)
            emb_org = torch.from_numpy(emb[np.newaxis, :].astype('float32')).to(self.device)
            # emb_trg = torch.from_numpy(emb[np.newaxis, :].astype('float32')).to(device)
            f0_org = torch.from_numpy(f0_org[np.newaxis, :].astype('float32')).to(self.device)
#             print('source shape:', x_real.shape, emb_org.shape, emb_trg.shape, f0_org.shape)

            with torch.no_grad():
                x_identic, x_identic_psnt_i, code_real = self.G(x_real, emb_org, f0_org, emb_trg, f0_org)
                x_identic_psnt.append(x_identic_psnt_i)

        x_identic_psnt = torch.cat(x_identic_psnt, dim=1)
#         print('converted shape:', x_identic_psnt.shape, code_real.shape)

        if len_pad == 0:
            uttr_trg = x_identic_psnt[0, :, :].cpu().numpy()
        else:
            uttr_trg = x_identic_psnt[0, :-len_pad, :].cpu().numpy()

        aus.append((uttr_trg, (0, audio_filename, emb)))

        return aus