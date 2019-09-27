# Copyright (c) 2019 NVIDIA Corporation
import nemo
import nemo_nlp
from nemo_nlp.callbacks.translation import eval_iter_callback, \
    eval_epochs_done_callback_wer
from nemo.core.callbacks import CheckpointCallback
from nemo.utils.lr_policies import CosineAnnealing

parser = nemo.utils.NemoArgParser(description='ASR postprocessor')
parser.set_defaults(
    train_dataset="train_8192",
    eval_datasets=["dev_clean", "dev_other", "test_clean", "test_other"],
    work_dir="asr_correction",
    optimizer="novograd",
    num_epochs=300,
    batch_size=4096,
    eval_batch_size=1024,
    lr=0.02,
    beta1=0.95,
    beta2=0.25,
    weight_decay=0,
    max_steps=300000,
    iter_per_step=1,
    checkpoint_save_freq=10000,
    eval_freq=5000)
parser.add_argument("--warmup_steps", default=4000, type=int)
parser.add_argument("--d_model", default=512, type=int)
parser.add_argument("--d_embedding", default=512, type=int)
parser.add_argument("--d_inner", default=2048, type=int)
parser.add_argument("--num_layers", default=6, type=int)
parser.add_argument("--num_attn_heads", default=8, type=int)
parser.add_argument("--embedding_dropout", default=0.1, type=float)
parser.add_argument("--ffn_dropout", default=0.1, type=float)
parser.add_argument("--attn_score_dropout", default=0.1, type=float)
parser.add_argument("--attn_layer_dropout", default=0.1, type=float)
parser.add_argument("--data_root", default="/dataset/", type=str)
parser.add_argument("--src_lang", default="pred", type=str)
parser.add_argument("--tgt_lang", default="real", type=str)
parser.add_argument("--beam_size", default=4, type=int)
parser.add_argument("--len_pen", default=0.0, type=float)
parser.add_argument("--tokenizer_model", default="m_common_8192.model", type=str)
parser.add_argument("--label_smoothing", default=0.1, type=float)
parser.add_argument("--tie_enc_dec", action="store_true")
parser.add_argument("--tie_enc_softmax", action="store_true")
parser.add_argument("--tie_projs", action="store_true")
parser.add_argument("--share_encoder_layers", action="store_true")
parser.add_argument("--share_decoder_layers", action="store_true")
parser.add_argument("--fp16", default=2, type=int)
args = parser.parse_args()

# Start Tensorboard X for logging
tb_name = "asr_postprocessor-lr{0}-opt{1}-warmup{2}-{3}-bs{4}".format(
    args.lr, args.optimizer, args.warmup_steps, "poly", args.batch_size)

if args.fp16 == 2:
    opt_level = nemo.core.Optimization.mxprO2
elif args.fp16 == 1:
    opt_level = nemo.core.Optimization.mxprO1
else:
    opt_level = nemo.core.Optimization.mxprO0

neural_factory = nemo.core.NeuralModuleFactory(
    local_rank=args.local_rank,
    optimization_level=opt_level,
    log_dir=args.work_dir,
    tensorboard_dir=tb_name,
)

tokenizer = nemo_nlp.YouTokenToMeTokenizer(
    model_path=f"{args.data_root}/{args.tokenizer_model}")
vocab_size = tokenizer.vocab_size

max_sequence_length = 512

train_data_layer = nemo_nlp.TranslationDataLayer(
    factory=neural_factory,
    tokenizer_src=tokenizer,
    tokenizer_tgt=tokenizer,
    dataset_src=args.data_root + args.train_dataset + "." + args.src_lang,
    dataset_tgt=args.data_root + args.train_dataset + "." + args.tgt_lang,
    tokens_in_batch=args.batch_size,
    clean=True)

eval_data_layers = {}

for key in args.eval_datasets:
    eval_data_layers[key] = nemo_nlp.TranslationDataLayer(
        factory=neural_factory,
        tokenizer_src=tokenizer,
        tokenizer_tgt=tokenizer,
        dataset_src=args.data_root + key + "." + args.src_lang,
        dataset_tgt=args.data_root + key + "." + args.tgt_lang,
        tokens_in_batch=args.eval_batch_size,
        clean=False)

encoder = nemo_nlp.TransformerEncoderNM(
    factory=neural_factory,
    d_embedding=args.d_embedding,
    d_model=args.d_model,
    d_inner=args.d_inner,
    num_layers=args.num_layers,
    num_attn_heads=args.num_attn_heads,
    ffn_dropout=args.ffn_dropout,
    vocab_size=vocab_size,
    max_seq_length=max_sequence_length,
    embedding_dropout=args.embedding_dropout,
    share_all_layers=args.share_encoder_layers,
    hidden_act="gelu")

decoder = nemo_nlp.TransformerDecoderNM(
    factory=neural_factory,
    d_embedding=args.d_embedding,
    d_model=args.d_model,
    d_inner=args.d_inner,
    num_layers=args.num_layers,
    num_attn_heads=args.num_attn_heads,
    ffn_dropout=args.ffn_dropout,
    vocab_size=vocab_size,
    max_seq_length=max_sequence_length,
    embedding_dropout=args.embedding_dropout,
    share_all_layers=args.share_encoder_layers,
    hidden_act="gelu")

log_softmax = nemo_nlp.TransformerLogSoftmaxNM(
    factory=neural_factory,
    vocab_size=vocab_size,
    d_model=args.d_model,
    d_embedding=args.d_embedding)

beam_translator = nemo_nlp.BeamSearchTranslatorNM(
    factory=neural_factory,
    decoder=decoder,
    log_softmax=log_softmax,
    max_seq_length=max_sequence_length,
    beam_size=args.beam_size,
    length_penalty=args.len_pen,
    bos_token=tokenizer.bos_id(),
    pad_token=tokenizer.pad_id(),
    eos_token=tokenizer.eos_id())

loss = nemo_nlp.PaddedSmoothedCrossEntropyLossNM(
    factory=neural_factory,
    pad_id=tokenizer.pad_id(),
    smoothing=0.1)

loss_eval = nemo_nlp.PaddedSmoothedCrossEntropyLossNM(
    factory=neural_factory,
    pad_id=tokenizer.pad_id(),
    smoothing=0.0)

# tie weight of embedding and log_softmax layers
if args.tie_enc_dec:
    decoder.embedding_layer.token_embedding.weight = \
        encoder.embedding_layer.token_embedding.weight
    if args.tie_projs:
        decoder.embedding_layer.token2hidden.weight = \
            encoder.embedding_layer.token2hidden.weight

if args.tie_enc_softmax:
    log_softmax.log_softmax.dense.weight = \
        encoder.embedding_layer.token_embedding.weight
    if args.tie_projs:
        log_softmax.log_softmax.hidden2token.weight = \
            encoder.embedding_layer.token2hidden.weight

# training pipeline
src, src_mask, tgt, tgt_mask, labels, sent_ids = train_data_layer()
src_hiddens = encoder(input_ids=src, input_mask_src=src_mask)
tgt_hiddens = decoder(input_ids_tgt=tgt,
                      hidden_states_src=src_hiddens,
                      input_mask_src=src_mask,
                      input_mask_tgt=tgt_mask)
log_probs = log_softmax(hidden_states=tgt_hiddens)
train_loss = loss(log_probs=log_probs, target_ids=labels)

# evaluation pipelines
src_ = {}
src_mask_ = {}
tgt_ = {}
tgt_mask_ = {}
labels_ = {}
sent_ids_ = {}
input_type_ids_ = {}
src_hiddens_ = {}
tgt_hiddens_ = {}
log_probs_ = {}
eval_loss_ = {}
beam_trans_ = {}

for key in args.eval_datasets:
    src_[key], src_mask_[key], tgt_[key], tgt_mask_[key], \
        labels_[key], sent_ids_[key] = eval_data_layers[key]()
    src_hiddens_[key] = encoder(
        input_ids=src_[key], input_mask_src=src_mask_[key])
    tgt_hiddens_[key] = decoder(
        input_ids_tgt=tgt_[key],
        hidden_states_src=src_hiddens_[key],
        input_mask_src=src_mask_[key],
        input_mask_tgt=tgt_mask_[key])
    log_probs_[key] = log_softmax(hidden_states=tgt_hiddens_[key])
    eval_loss_[key] = loss_eval(
        log_probs=log_probs_[key],
        target_ids=labels_[key])
    beam_trans_[key] = beam_translator(
        hidden_states_src=src_hiddens_[key],
        input_mask_src=src_mask_[key])


def print_loss(x):
    loss = x[0].item()
    neural_factory.logger.info("Training loss: {:.4f}".format(loss))


# Create evaluation callbacks
callback_train = nemo.core.SimpleLossLoggerCallback(
    tensors=[train_loss],
    step_freq=100,
    print_func=print_loss,
    get_tb_values=lambda x: [["loss", x[0]]],
    tb_writer=neural_factory.tb_writer)

callbacks = [callback_train]

for key in args.eval_datasets:

    callback = nemo.core.EvaluatorCallback(
        eval_tensors=[
            tgt_[key], eval_loss_[key], beam_trans_[key], sent_ids_[key]
        ],
        user_iter_callback=lambda x, y: eval_iter_callback(x, y, tokenizer),
        user_epochs_done_callback=eval_epochs_done_callback_wer,
        eval_step=args.eval_freq,
        tb_writer=neural_factory.tb_writer)

    callbacks.append(callback)

checkpointer_callback = CheckpointCallback(
    folder=args.work_dir, step_freq=args.checkpoint_save_freq)
callbacks.append(checkpointer_callback)

# define learning rate decay policy
lr_policy = CosineAnnealing(args.max_steps, warmup_steps=args.warmup_steps)

# Create trainer and execute training action
neural_factory.train(
    tensors_to_optimize=[train_loss],
    callbacks=callbacks,
    optimizer=args.optimizer,
    lr_policy=lr_policy,
    optimization_params={
        "num_epochs": args.num_epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "betas": (0.95, 0.25)},
    batches_per_step=args.iter_per_step)
