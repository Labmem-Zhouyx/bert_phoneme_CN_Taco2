# coding:utf-8
import sys
import numpy as np
import torch
import os
import argparse
import json
import codecs

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from hparams import hparams
from models import Semantic_Tacotron2
from transformers import BertTokenizer
from distributed import apply_gradient_allreduce


class TextMelLoaderEval(torch.utils.data.Dataset):
    def __init__(self, sentences, hparams):
        self.sentences = sentences

        with codecs.open(hparams.class2idx, 'r', 'utf-8') as usernames:
            self.class2idx = json.load(usernames)

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")


    def text_to_sequence(self, text):
        # append an EOS token
        text = text.strip()
        char_text = text.split('|')[0]
        phoneme_text = text.split('|')[1]
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
        return len(self.sentences)

    def __getitem__(self, index):
        return self.text_to_sequence(self.sentences[index])


class TextMelCollateEval():
    """ Zero-pads model inputs and targets based on number of frames per setep
    """
    def __init__(self, hparams):
        self.n_frames_per_step = hparams.n_frames_per_step

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

        return input_lengths, inputs_padded, phonemes_padded


def get_sentences(args):
    if args.text_file != '':
        with open(args.text_file, 'rb') as f:
            sentences = list(map(lambda l: l.decode("utf-8")[:-1], f.readlines()))
    else:
        sentences = [args.sentences]
    print("Check sentences:", sentences)
    return sentences


def load_model(hparams):
    model = Semantic_Tacotron2(hparams).cuda()
    if hparams.fp16_run:
        model.decoder.attention_layer.score_mask_value = finfo('float16').min

    if hparams.distributed_run:
        model = apply_gradient_allreduce(model)

    return model

def inference(args):

    sentences = get_sentences(args)

    model = load_model(hparams)
    model.load_state_dict(torch.load(args.checkpoint)['state_dict'])
    model.cuda().eval()#.half()

    test_set = TextMelLoaderEval(sentences, hparams)
    test_collate_fn = TextMelCollateEval(hparams)
    test_sampler = DistributedSampler(valset) if hparams.distributed_run else None
    test_loader = DataLoader(test_set, num_workers=0, sampler=test_sampler, batch_size=hparams.synth_batch_size, pin_memory=False, drop_last=True, collate_fn=test_collate_fn)

    T2_output_range = (-hparams.max_abs_value, hparams.max_abs_value) if hparams.symmetric_mels else (0, hparams.max_abs_value)

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            mel_outputs, mel_outputs_postnet, _, alignments = model.inference(batch)

            mels = mel_outputs_postnet[0].cpu().numpy()

            mel_path = os.path.join(args.out_filename, 'sentence_{}_mel-feats.npy'.format(i))
            mels = np.clip(mels, T2_output_range[0], T2_output_range[1])
            np.save(mel_path, mels.T, allow_pickle=False)

            print('CHECK MEL SHAPE:', mels.T.shape)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--sentences', type=str, help='text to infer', default='啊这|a1 zhe4')
    parser.add_argument('-t', '--text_file', type=str, help='text file to infer', default='./sentences.txt')
    parser.add_argument('-c', '--checkpoint', type=str, help='checkpoint path',
                       default='./test_bert_phoneme_taco2/checkpoint_10000')

    parser.add_argument('-o', '--out_filename', type=str, help='output filename', default='./inference_mel')
    args = parser.parse_args()
    inference(args)