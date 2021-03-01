import json
import random
import codecs
import numpy as np
import torch
import torch.utils.data

from utils import load_filepaths_and_text

from transformers import BertTokenizer
from pytorch_pretrained_bert import BertModel


class TextMelLoader(torch.utils.data.Dataset):
    """
        1) loads audio,text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
    """
    def __init__(self, melpaths_and_text, hparams):
        self.melpaths_and_text = load_filepaths_and_text(melpaths_and_text)
        self.text_cleaners = hparams.text_cleaners

        with codecs.open(hparams.class2idx, 'r', 'utf-8') as usernames:
            self.class2idx = json.load(usernames)

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

        random.seed(hparams.seed)
        random.shuffle(self.melpaths_and_text)

    def get_mel_text_pair(self, melpath_and_text):
        # separate filename and text
        melpath, char_text, phoneme_text = melpath_and_text[0], melpath_and_text[1], melpath_and_text[2]
        char_seq, phoneme_seq = self.text_to_sequence(char_text, phoneme_text)
        mel = torch.from_numpy(np.load(melpath))

        return (char_seq, phoneme_seq, mel)

    def text_to_sequence(self, char_text, phoneme_text):
        # append an EOS token
        toks = self.tokenizer.tokenize(char_text + '~')
        pinyins = phoneme_text.strip().split(' ')
        pinyins.append('~')
        assert len(toks) == len(pinyins)
        char_ids = self.tokenizer.convert_tokens_to_ids(toks)
        char_ids = torch.tensor(char_ids, dtype=torch.long)
        phoneme_ids = [self.class2idx[i] for i in pinyins]
        phoneme_ids = torch.tensor(phoneme_ids, dtype=torch.long)
        return char_ids, phoneme_ids

    def __len__(self):
        return len(self.melpaths_and_text)

    def __getitem__(self, index):
        return self.get_mel_text_pair(self.melpaths_and_text[index])


class TextMelCollate():
    """ Zero-pads model inputs and targets based on number of frames per setep
    """
    def __init__(self, n_frames_per_step):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        inputs_padded = torch.LongTensor(len(batch), max_input_len)
        inputs_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            input_id = batch[ids_sorted_decreasing[i]][0]
            inputs_padded[i, :input_id.shape[0]] = input_id

        phonemes_padded = torch.LongTensor(len(batch), max_input_len)
        phonemes_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            phoneme_id = batch[ids_sorted_decreasing[i]][1]
            phonemes_padded[i, :phoneme_id.shape[0]] = phoneme_id

        # Right zero-pad mel-spec
        num_mels = batch[0][2].size(0)
        max_target_len = max([x[2].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded and gate padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][2]
            mel_padded[i, :, :mel.size(1)] = mel
            gate_padded[i, mel.size(1)-1:] = 1
            output_lengths[i] = mel.size(1)

        return input_lengths, inputs_padded, phonemes_padded, mel_padded, gate_padded, output_lengths