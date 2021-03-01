import os
import json
import random
import codecs
# import parselmouth
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
from math import floor
from .pinyin import split_pinyin
# from datasets import audio
# from wavenet_vocoder.util import is_mulaw, is_mulaw_quantize, mulaw, mulaw_quantize


def is_chinese(uchar):
    """判断一个unicode是否是汉字"""
    if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
        return True
    else:
        return False


def calculate_durations(interval_path, hparams):
    durs = []
    durs_name = []
    durs_original = []
    line_id = 0
    phone_num = 0
    hop_time = hparams.hop_size / hparams.sample_rate
    with open(interval_path, encoding='utf-8', errors='ignore') as f:
        for line in f:
            line_id = line_id + 1
            if line_id == 12:
                phone_num = line
            if line_id > 12:
                begin_time = line
                end_time = f.readline()
                line_id = line_id + 1
                phone_name = f.readline()
                line_id = line_id + 1
                durs_original.append([begin_time, end_time, phone_name.strip()])
    for dur in durs_original:
        dur_time = float(dur[1]) - float(dur[0])
        dur_length = floor(dur_time / hop_time)
        durs.append(dur_length)
        durs_name.append(dur[2])
    return durs, durs_name


def dur_chunk_sizes(n, ary):
    """Split a single duration into almost-equally-sized chunks

    Examples:
      dur_chunk(3, 2) --> [2, 1]
      dur_chunk(3, 3) --> [1, 1, 1]
      dur_chunk(5, 3) --> [2, 2, 1]
    """
    ret = np.ones((ary,), dtype=np.int32) * (n // ary)
    ret[:n % ary] = n // ary + 1
    assert ret.sum() == n
    return ret


def maybe_pad(vec, l):
    assert np.abs(vec.shape[0] - l) <= 3
    vec = vec[:l]
    if vec.shape[0] < l:
        vec = np.pad(vec, pad_width=(0, l - vec.shape[0]))
    return vec


def calculate_pitch(wav, durs):
    # mel_len = durs.sum()
    mel_len = sum(durs)
    durs_cum = np.cumsum(np.pad(durs, (1, 0)))
    # snd = parselmouth.Sound(wav)
    pitch = snd.to_pitch(time_step=snd.duration / (mel_len + 3)
                         ).selected_array['frequency']
    # assert np.abs(mel_len - pitch.shape[0]) <= 1.0
    if np.abs(mel_len - pitch.shape[0]) > 1.0:
        print('CHECK not assert wav path:', wav)

    # Average pitch over characters
    # pitch_char = np.zeros((durs.shape[0],), dtype=np.float)
    pitch_char = np.zeros((len(durs),), dtype=np.float)
    for idx, a, b in zip(range(mel_len), durs_cum[:-1], durs_cum[1:]):
        values = pitch[a:b][np.where(pitch[a:b] != 0.0)[0]]
        pitch_char[idx] = np.mean(values) if len(values) > 0 else 0.0

    # Average to three values per character
    # pitch_trichar = np.zeros((3 * durs.shape[0],), dtype=np.float)
    pitch_trichar = np.zeros((3 * len(durs),), dtype=np.float)

    durs_tri = np.concatenate([dur_chunk_sizes(d, 3) for d in durs])
    durs_tri_cum = np.cumsum(np.pad(durs_tri, (1, 0)))

    for idx, a, b in zip(range(3 * mel_len), durs_tri_cum[:-1], durs_tri_cum[1:]):
        values = pitch[a:b][np.where(pitch[a:b] != 0.0)[0]]
        pitch_trichar[idx] = np.mean(values) if len(values) > 0 else 0.0

    pitch_mel = maybe_pad(pitch, mel_len)
    pitch_char = maybe_pad(pitch_char, len(durs))
    pitch_trichar = maybe_pad(pitch_trichar, len(durs_tri))

    return pitch_mel, pitch_char, pitch_trichar


def build_from_path(hparams, input_dirs, out_dir, mel_dir, linear_dir, pitch_dir, wav_dir, pinyin_symbols, n_jobs=12, tqdm=lambda x: x):
	"""
	Preprocesses the speech dataset from a gven input path to given output directories

	Args:
		- hparams: hyper parameters
		- input_dir: input directory that contains the files to prerocess
		- mel_dir: output directory of the preprocessed speech mel-spectrogram dataset
		- linear_dir: output directory of the preprocessed speech linear-spectrogram dataset
		- wav_dir: output directory of the preprocessed speech audio dataset
		- n_jobs: Optional, number of worker process to parallelize across
		- tqdm: Optional, provides a nice progress bar

	Returns:
		- A list of tuple describing the train examples. this should be written to train.txt
	"""

	# We use ProcessPoolExecutor to parallelize across processes, this is just for
	# optimization purposes and it can be omited
	executor = ProcessPoolExecutor(max_workers=n_jobs)
	futures = []
	characters = []
	pinyins = []
	polyphone_dict = {}
	# pitch_dict = {}
	index = 1
	for input_dir in input_dirs:
		with open(os.path.join(input_dir, 'textFromBZNSYP.txt'), encoding='utf-8', errors='ignore') as f:
			for line in f:
				line1 = line
				line_id = line1.strip().split('\t')[0]

				line_text = line1.strip().split('\t')[1]
				character = [w for w in line_text if is_chinese(w)]
				# print('CHECK character:', character)
				characters.extend(character)
				line2 = f.readline()
				if not line2:
					break

				if line_id == '002365':
					print('No 002365')
					continue

				er_phoneme = ['er1', 'er2', 'er3', 'er4', 'er5', 'rr1', 'rr2', 'rr3', 'rr4', 'rr5']
				pinyin_no_er_phoneme = []
				pinyin = line2.strip().split(' ')
				for pinyin_to_delect_er in pinyin:
					# print('CHECK pinyin_to_delect_er:', pinyin_to_delect_er)
					if pinyin_to_delect_er in er_phoneme:
						pinyin_no_er_phoneme.append(pinyin_to_delect_er)
					else:
						if pinyin_to_delect_er[-2] == 'r':
							print('CHECK HERE ORIGINAL:', pinyin_to_delect_er)
							pinyin_temp = pinyin_to_delect_er[:-2] + pinyin_to_delect_er[-1]
							# pinyin_temp = ''.join(list(pinyin_to_delect_er).pop(-2))
							pinyin_no_er_phoneme.append(pinyin_temp)
							print('CHECK HERE AFTER:', pinyin_temp)
						else:
							pinyin_no_er_phoneme.append(pinyin_to_delect_er)
				pinyins.extend(pinyin_no_er_phoneme)
				# print('CHECK pinyin:', pinyin)
				# assert len(pinyin) == len(character)
				if not len(pinyin) == len(character):

					while '儿' in character:
						character.remove('儿')
					if not len(pinyin) == len(character):
						print('ER HUA YIN HERE!')
						print('CHECK character:', character)
						print('CHECK pinyin:', pinyin)
						continue

				for one_char, one_pinyin in zip(character, pinyin):
					if one_char in polyphone_dict.keys():
						polyphone_dict[one_char].append(one_pinyin)
					else:
						polyphone_dict[one_char] = [one_pinyin]

				assert len(character) == len(pinyin)
				text = "".join(character)
				# print('CHECK text:', text)
				pinyin_string = " ".join(pinyin)
				# print('CHECK pinyin_string:', pinyin_string)

				# text = line2.strip().split('\t')[0]
				# print('TEXTS:', text)

				# if index > 100:
				# 	break

				# parts = line.strip().split('|')
				# basename = parts[0]
				# wav_path = os.path.join(input_dir, 'wavs', '{}.wav'.format(basename))
				wav_path = os.path.join(input_dir, 'Wave-16k', '%s.wav' % line_id)
				duration_path = os.path.join(input_dir, 'PhoneLabeling', '%s.interval' % line_id)
				# text = parts[2]
				# futures.append(executor.submit(partial(_process_utterance, pitch_dict, mel_dir, linear_dir, pitch_dir, wav_dir, line_id, duration_path, wav_path, text, hparams)))
				futures_to_append = _process_utterance(mel_dir, linear_dir, wav_dir, line_id, duration_path, wav_path, text, pinyin_string, hparams)
				futures.append(futures_to_append)
				index += 1
	
	# print('CHECK characters:', list(set(characters)))
	# print('CHECK characters length:', len(list(set(characters))))
	# pinyin_symbles = list(set(pinyins))
	# print('CHECK pinyins:', pinyin_symbles)
	# print('CHECK pinyins length:', len(pinyin_symbles))

	# print('CHECK pitch_dict:', pitch_dict)
	# polyphone_dict_filename = os.path.join(out_dir, 'polyphone_dict.json')
	# polyphone_dict_set = {}
	# polyphone_length = []

	# padding_max_length = 6
	# padding_pinyin = 'xxx'

	# polyphone_dict_toshow = {}

	# for (k,v) in  polyphone_dict.items():
		# # print('CHECK character', k)
		# v = list(set(v))
		# # padding with and shuffle\
		# poly_list = v
		# poly_list += [padding_pinyin for i in range(padding_max_length-len(poly_list))]
		# random.shuffle(poly_list)
		# # print('CHECK poly original:', v)
		# # print('CHECK poly original:', poly_list)
		# # print('CHECK character length', len(v))
		# if len(v) > 1:
			# polyphone_length.append(len(v))
			# polyphone_dict_toshow[k] = v
		# # temp = np.array(v)
		# # arr_mean = np.mean(temp)
		# # arr_std = np.std(temp, ddof=1)
		# polyphone_dict_set[k] = poly_list
	# # print('CHECK pitch_dict_per_phoneme:', pitch_dict_per_phoneme)
	# polyphone_length_numpy = np.array(polyphone_length)
	# minNum = np.mean(polyphone_length_numpy)
	# print('CHECK MIN NUM:', minNum)
	# maxNum = np.max(polyphone_length_numpy)
	# print('CHECK MAX NUM:', maxNum)


	polyphone_dict_filename = os.path.join(out_dir, 'polyphone_mask.json')
	polyphone_dict_set = {}
	# polyphone_length = []

	padding_max_length = 6
	padding_pinyin = 'xxx'

	# polyphone_dict_toshow = {}

	for (k,v) in  polyphone_dict.items():
		print('CHECK character', k)
		v = list(set(v))
		print('CHECK character value in polyphone_dict', v)
		# padding with and shuffle\
		poly_mask = {}
		for _one_v in v:

			if _one_v in er_phoneme:
				_one_v = _one_v
			else:
				if _one_v[-2] == 'r':
					_one_v = _one_v[:-2] + _one_v[-1]
				else:
					_one_v = _one_v


			one_mask = pinyin_symbols.index(_one_v)
			poly_mask[_one_v] = one_mask
		# poly_list = v
		# poly_list += [padding_pinyin for i in range(padding_max_length-len(poly_list))]
		# random.shuffle(poly_list)
		# print('CHECK poly original:', v)
		# print('CHECK poly original:', poly_list)
		# print('CHECK character length', len(v))
		# if len(v) > 1:
			# polyphone_length.append(len(v))
			# polyphone_dict_toshow[k] = v
		# temp = np.array(v)
		# arr_mean = np.mean(temp)
		# arr_std = np.std(temp, ddof=1)
		polyphone_dict_set[k] = poly_mask
	# print('CHECK pitch_dict_per_phoneme:', pitch_dict_per_phoneme)
	# polyphone_length_numpy = np.array(polyphone_length)
	# minNum = np.mean(polyphone_length_numpy)
	# print('CHECK MIN NUM:', minNum)
	# maxNum = np.max(polyphone_length_numpy)
	# print('CHECK MAX NUM:', maxNum)

	print('CHECK ALL polyphone_dict_toshow', polyphone_dict_set)
	print('CHECK ALL polyphone_dict_toshow', len(polyphone_dict_set))

	with codecs.open(polyphone_dict_filename, 'w', 'utf-8') as usernames:
		json.dump(polyphone_dict_set, usernames, ensure_ascii=False)

	# return [future.result() for future in tqdm(futures) if future.result() is not None]
	return [future for future in tqdm(futures) if future is not None]


def _process_utterance(mel_dir, linear_dir, wav_dir, index, duration_path, wav_path, text, pinyin_string, hparams):
	"""
	Preprocesses a single utterance wav/text pair

	this writes the mel scale spectogram to disk and return a tuple to write
	to the train.txt file

	Args:
		- mel_dir: the directory to write the mel spectograms into
		- linear_dir: the directory to write the linear spectrograms into
		- wav_dir: the directory to write the preprocessed wav into
		- index: the numeric index to use in the spectogram filename
		- wav_path: path to the audio file containing the speech input
		- text: text spoken in the input audio file
		- hparams: hyper parameters

	Returns:
		- A tuple: (audio_filename, mel_filename, linear_filename, time_steps, mel_frames, linear_frames, text)
	"""
	# try:
		# # Load the audio as numpy array
		# wav = audio.load_wav(wav_path, sr=hparams.sample_rate)
	# except FileNotFoundError: #catch missing wav exception
		# print('file {} present in csv metadata is not present in wav folder. skipping!'.format(
			# wav_path))
		# return None


	# durs, durs_name = calculate_durations(duration_path, hparams)
	# # print('CHECK durs of:', index, sum(durs))
	# # print('CHECK durs_name:', durs_name)
	# # assert len(durs) == len(durs_name)
	# ## pitch_mel, pitch_char, pitch_trichar = calculate_pitch(wav_path, durs)
	# # print('CHECK pitch_mel:', pitch_mel)
	# # print('CHECK pitch_char:', pitch_char)
	# # print('CHECK pitch_trichar:', pitch_trichar)
	# # print('CHECK pitch_mel length:', index, len(pitch_mel))

	# durs_name_str = [name.replace('"', '') for name in durs_name]
	# durs_name_string = " ".join(durs_name_str)
	# # print('CHECK durs_name_string:', durs_name_string)


	# for name, pitch in zip(durs_name_str, pitch_char):
		# if name in pitch_dict.keys():
			# pitch_dict[name].append(pitch)
		# else:
			# pitch_dict[name] = [pitch]

	# # assert len(durs) == len(pitch_char)
	# # assert sum(durs) == len(pitch_mel)

	# #Trim lead/trail silences
	# if hparams.trim_silence:
		# wav = audio.trim_silence(wav, hparams)

	# #Pre-emphasize
	# preem_wav = audio.preemphasis(wav, hparams.preemphasis, hparams.preemphasize)

	# #rescale wav
	# if hparams.rescale:
		# wav = wav / np.abs(wav).max() * hparams.rescaling_max
		# preem_wav = preem_wav / np.abs(preem_wav).max() * hparams.rescaling_max

		# #Assert all audio is in [-1, 1]
		# if (wav > 1.).any() or (wav < -1.).any():
			# raise RuntimeError('wav has invalid value: {}'.format(wav_path))
		# if (preem_wav > 1.).any() or (preem_wav < -1.).any():
			# raise RuntimeError('wav has invalid value: {}'.format(wav_path))

	# #Mu-law quantize
	# if is_mulaw_quantize(hparams.input_type):
		# #[0, quantize_channels)
		# out = mulaw_quantize(wav, hparams.quantize_channels)

		# #Trim silences
		# start, end = audio.start_and_end_indices(out, hparams.silence_threshold)
		# wav = wav[start: end]
		# preem_wav = preem_wav[start: end]
		# out = out[start: end]

		# constant_values = mulaw_quantize(0, hparams.quantize_channels)
		# out_dtype = np.int16

	# elif is_mulaw(hparams.input_type):
		# #[-1, 1]
		# out = mulaw(wav, hparams.quantize_channels)
		# constant_values = mulaw(0., hparams.quantize_channels)
		# out_dtype = np.float32

	# else:
		# #[-1, 1]
		# out = wav
		# constant_values = 0.
		# out_dtype = np.float32

	# # Compute the mel scale spectrogram from the wav
	# mel_spectrogram = audio.melspectrogram(preem_wav, hparams).astype(np.float32)
	# mel_frames = mel_spectrogram.shape[1]

	# if mel_frames > hparams.max_mel_frames and hparams.clip_mels_length:
		# return None

	# #Compute the linear scale spectrogram from the wav
	# linear_spectrogram = audio.linearspectrogram(preem_wav, hparams).astype(np.float32)
	# linear_frames = linear_spectrogram.shape[1]

	# #sanity check
	# assert linear_frames == mel_frames
	# # print('CHECK frames num of:', index, linear_frames)
	# # assert linear_frames == len(pitch_mel)

	# if hparams.use_lws:
		# #Ensure time resolution adjustement between audio and mel-spectrogram
		# fft_size = hparams.n_fft if hparams.win_size is None else hparams.win_size
		# l, r = audio.pad_lr(wav, fft_size, audio.get_hop_size(hparams))

		# #Zero pad audio signal
		# out = np.pad(out, (l, r), mode='constant', constant_values=constant_values)
	# else:
		# #Ensure time resolution adjustement between audio and mel-spectrogram
		# l_pad, r_pad = audio.librosa_pad_lr(wav, hparams.n_fft, audio.get_hop_size(hparams), hparams.wavenet_pad_sides)

		# #Reflect pad audio signal on the right (Just like it's done in Librosa to avoid frame inconsistency)
		# out = np.pad(out, (l_pad, r_pad), mode='constant', constant_values=constant_values)

	# assert len(out) >= mel_frames * audio.get_hop_size(hparams)

	# #time resolution adjustement
	# #ensure length of raw audio is multiple of hop size so that we can use
	# #transposed convolution to upsample
	# out = out[:mel_frames * audio.get_hop_size(hparams)]
	# assert len(out) % audio.get_hop_size(hparams) == 0
	# time_steps = len(out)

	# # Write the spectrogram and audio to disk
	# audio_filename = 'audio-{}.npy'.format(index)
	# mel_filename = 'mel-{}.npy'.format(index)
	# linear_filename = 'linear-{}.npy'.format(index)
	# pitch_filename = 'pitch-{}.npy'.format(index)
	# np.save(os.path.join(wav_dir, audio_filename), out.astype(out_dtype), allow_pickle=False)
	# np.save(os.path.join(mel_dir, mel_filename), mel_spectrogram.T, allow_pickle=False)
	# np.save(os.path.join(linear_dir, linear_filename), linear_spectrogram.T, allow_pickle=False)
	# np.save(os.path.join(pitch_dir, pitch_filename), np.array(pitch_char), allow_pickle=False)

	# Return a tuple describing this training example
	# return (audio_filename, mel_filename, linear_filename, time_steps, mel_frames, text, pitch_filename, durs_name_string), pitch_dict
	return (wav_path, text, pinyin_string)
