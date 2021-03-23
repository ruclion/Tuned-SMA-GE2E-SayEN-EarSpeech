import os
import preprocess_dataset.mel as mel
import numpy as np
from pathlib import Path
from synthesizer.preprocess import embed_utterance



# all meta path
# [1] in_path
metas_path = ['/ceph/home/hujk17/Tuned-GE2E-SayEN-EarSpeech/preprocess_dataset/train_meta_full_106_nosli.txt',
              '/ceph/home/hujk17/Tuned-GE2E-SayEN-EarSpeech/preprocess_dataset/val_meta_full_106_nosli.txt',
              '/ceph/home/hujk17/Tuned-GE2E-SayEN-EarSpeech/preprocess_dataset/unseen_meta_full_106_nosli.txt']

# [2] in_wav_dir
wav_dir_path = '/ceph/home/hujk17/VCTK-Corpus/wav16_nosli'

# [3] in_txt_dir
txt_dir_path = '/ceph/dataset/VCTK-Corpus/txt'



# all out path
out_dir_path = '/ceph/home/hujk17/Tuned-GE2E-SayEN-EarSpeech/preprocess_dataset'
out_txt_path = ['train.txt', 'val.txt', 'unseen.txt']



def _process_utterance(audio_path, text, npy_path):
	if os.path.exists(audio_path):
		mel_spectrogram, _linear_spectrogram, out = mel.wav2mel(audio_path)
		time_steps = len(out)
		mel_frames = mel_spectrogram.shape[0]
	else:
		print('file {} present in csv metadata is not present in wav folder. skipping!'.format(audio_path))
		return None

	#    /ceph/home/hujk17/npy-EarSpeech-HCSI-Data/tst_npy/MST-Originbeat-S2-male-5000/spk-004996-GE2E.npy
	#->  /ceph/home/hujk17/npy-EarSpeech-HCSI-Data/tst_npy/MST-Originbeat-S2-male-5000/mel-004996.npy
	mel_same_with_npy_path = npy_path.replace('spk', 'mel').replace('-GE2E', '').split('.')[0] + '-mel.npy'
	np.save(mel_same_with_npy_path, mel_spectrogram.T, allow_pickle=False)
	# print(mel_same_with_npy_path)
	# print(mel_spectrogram.shape, mel_spectrogram)
	# Return a tuple describing this training example
	return (mel_same_with_npy_path, npy_path, time_steps, mel_frames, text)




def main():
    print('begin:', metas_path)
    for no in range(len(metas_path)):
        now_file = metas_path[no]
        out_file = os.path.join(out_dir_path, out_txt_path[no])
        # print(now_file, out_file)
        res = []
        with open(now_file, 'r', encoding='utf-8') as f:
            f_list = f.readlines()
            for x in f_list:
                # p227/p227_009.npy|p227
                x = x.strip().split('|')
                speaker_id = x[1]
                audio_id = x[0].split('/')[1].split('.')[0]

                # audio_path
                audio_path = os.path.join(wav_dir_path, os.path.join(speaker_id, audio_id + '.wav'))
                # print('now:', audio_path, os.path.exists(audio_path))
                if os.path.exists(audio_path) is False:
                    continue
                
                # txt
                txt_path = os.path.join(txt_dir_path, os.path.join(speaker_id, audio_id + '.txt'))
                if os.path.exists(txt_path) is False:
                    continue
                with open(txt_path) as sentence_f:
                    s_f_list = sentence_f.readlines()
                    txt = s_f_list[0].strip()
                # print('txt:', txt)

                # speaker_npy_path
                GE2E_npy = os.path.join(wav_dir_path, os.path.join(speaker_id, 'spk-' + audio_id + '-GE2E.npy'))
                embed_utterance((audio_path, GE2E_npy), encoder_model_fpath=Path('encoder/saved_models/pretrained.pt'))
                # print(GE2E_npy)

                # mel_npy_path
                t = _process_utterance(audio_path=audio_path, text=txt, npy_path=GE2E_npy)
                res.append(t)

                # break
        with open(out_file, 'w', encoding='utf-8') as f:
            for i in range(len(res)):
                f.write('|'.join([str(x) for x in res[i]]) + '\n')
        
        # break




if __name__ == '__main__':
    main()




