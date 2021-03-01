import argparse
import os
from multiprocessing import cpu_count

from hparams import hparams
from tqdm import tqdm
from datasets import ljspeech
from datasets import databaker
from datasets import multisets


def write_metadata(metadata, out_dir):
	with open(os.path.join(out_dir, 'train.txt'), 'w', encoding='utf-8') as f:
		for m in metadata:
			f.write('|'.join([str(x) for x in m]) + '\n')
	mel_frames = sum([int(m[4]) for m in metadata])
	timesteps = sum([int(m[3]) for m in metadata])
	sr = hparams.sampling_rate
	hours = timesteps / sr / 3600
	print('Write {} utterances, {} mel frames, {} audio timesteps, ({:.2f} hours)'.format(
		len(metadata), mel_frames, timesteps, hours))
	print('Max input length (text chars): {}'.format(max(len(m[5]) for m in metadata)))
	print('Max mel frames length: {}'.format(max(int(m[4]) for m in metadata)))
	print('Max audio timesteps length: {}'.format(max(m[3] for m in metadata)))


def main():
	print('initializing preprocessing..')
	parser = argparse.ArgumentParser()
	parser.add_argument('--base_dir', default='')
	parser.add_argument('--hparams', default='',
		help='Hyperparameter overrides as a comma-separated list of name=value pairs')
	parser.add_argument('--dataset', default='MultiSets')
	parser.add_argument('--output', default='training_data')
	parser.add_argument('--n_jobs', type=int, default=cpu_count())
	args = parser.parse_args()
	
	# Prepare directories
	# in_dir  = os.path.join(args.base_dir, args.dataset)
	# out_dir = os.path.join(args.base_dir, args.output)
	in_dir = args.base_dir
	out_dir = args.output
	mel_dir = os.path.join(out_dir, 'mels')
	wav_dir = os.path.join(out_dir, 'audio')
	lin_dir = os.path.join(out_dir, 'linear')
	os.makedirs(mel_dir, exist_ok=True)
	os.makedirs(wav_dir, exist_ok=True)
	os.makedirs(lin_dir, exist_ok=True)
	
	# Process dataset
	if args.dataset == 'LJSpeech-1.1':
		metadata = ljspeech.build_from_path(hparams, in_dir, mel_dir, lin_dir, wav_dir, args.n_jobs, tqdm=tqdm)
	elif args.dataset == 'DataBaker':
		use_prosody = False
		metadata = databaker.build_from_path(hparams, in_dir, use_prosody, mel_dir, lin_dir, wav_dir, args.n_jobs, tqdm=tqdm)
	elif args.dataset == 'MultiSets':
		metadata = multisets.build_from_path(hparams, in_dir, mel_dir, lin_dir, wav_dir, args.n_jobs, tqdm=tqdm)
	else:
		raise ValueError('Unsupported dataset provided: {} '.format(args.dataset))
	
	# Write metadata to 'train.txt' for training
	write_metadata(metadata, out_dir)


if __name__ == '__main__':
	main()
