from text import yinsu_symbols, character_symbols, pinyin_symbols


class hparams:
        ################################
        # Experiment Parameters        #
        ################################
        epochs=500
        iters_per_checkpoint=5000
        seed=4321
        dynamic_loss_scaling=True
        fp16_run=False
        distributed_run=False
        dist_backend="nccl"
        dist_url="tcp://localhost:54321"
        cudnn_enabled=True
        cudnn_benchmark=False
        ignore_layers=['embedding.weight']
        synth_batch_size = 1
        ################################
        # Data Parameters             #
        ################################
        load_mel_from_disk=True
        mel_training_files='./training_data/mel-bznsyp_character_pinyin_data_train.txt'
        mel_validation_files='./training_data/mel-bznsyp_character_pinyin_data_val.txt'
        class2idx = "./training_data/bznsyp_pinyin2idx.json"
        text_cleaners=['basic_cleaners']

        ################################
        # Audio Parameters             #
        ################################
        max_wav_value=32768.0
        sampling_rate=16000
        filter_length=1024
        hop_length = 200
        win_length = 800
        n_mel_channels=80
        mel_fmin=0.0
        mel_fmax=8000.0
        max_abs_value = 4.0
        symmetric_mels = True

        ################################
        # Model Parameters             #
        ################################
        phoneme_num=len(yinsu_symbols)
        n_character_symbols=len(character_symbols)
        pinyin_num=1736
        bert_dim=768
        character_embedding_dim=512
        phoneme_embedding_dim=512
        # Encoder parameters
        encoder_kernel_size=5
        encoder_n_convolutions=3
        encoder_embedding_dim=512     # 512
        encoder_input_dim = [512 + 768, 512, 512]
        encoder_output_dim=[512, 512, 512]

        # Decoder parameters
        n_frames_per_step=3
        decoder_rnn_dim=1024
        prenet_dim=256
        max_decoder_steps=1000
        gate_threshold=0.5
        p_attention_dropout=0.1
        p_decoder_dropout=0.1

        # Attention parameters
        attention_rnn_dim=1024
        attention_dim=128

        # Location Layer parameters
        attention_location_n_filters=32
        attention_location_kernel_size=31

        # Mel-post processing network parameters
        postnet_embedding_dim=512
        postnet_kernel_size=5
        postnet_n_convolutions=5

        ################################
        # Optimization Hyperparameters #
        ################################
        use_saved_learning_rate=False
        learning_rate=1e-3
        weight_decay=1e-6
        grad_clip_thresh=1.0
        batch_size=32
        mask_padding=True  # set model's padded outputs to padded values

