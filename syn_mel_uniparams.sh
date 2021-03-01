#!/bin/bash

config_path=/ceph/home/zhouyx20/code/ParallelWaveGAN/egs/multi_spk_hcclspeech/MBMELGAN_uni_params.yaml
dumpdir=/ceph/home/zhouyx20/code/bert_phoneme_CN_Taco2/inference_mel/
outdir=/ceph/home/zhouyx20/code/bert_phoneme_CN_Taco2/inference_wav/
checkpoint=/ceph/home/zhouyx20/code/ParallelWaveGAN/egs/multi_spk_hcclspeech/model_mel_uniparams/checkpoint-1000000steps.pkl

CUDA_VISIBLE_DEVICES=2 python /ceph/home/zhouyx20/code/ParallelWaveGAN/parallel_wavegan/bin/decode.py --config ${config_path} --dumpdir ${dumpdir} --outdir ${outdir} --checkpoint ${checkpoint}