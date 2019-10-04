# Copyright (c) 2019 NVIDIA Corporation
import torch
import nemo
import math
import nemo_nlp
from nemo_nlp.callbacks.translation import eval_iter_callback, \
    eval_epochs_done_callback_wer
from nemo_nlp.data.tokenizers.bert_tokenizer import NemoBertTokenizer
from nemo.core.callbacks import CheckpointCallback
from nemo.utils.lr_policies import SquareAnnealing

parser = nemo.utils.NemoArgParser(description='ASR postprocessor')
parser.set_defaults(train_dataset="train",
                    eval_datasets=["dev_clean", "dev_other",
                                   "test_clean", "test_other"],
                    work_dir="asr_correction",
                    optimizer="novograd",
                    num_epochs=1000,
                    batch_size=4096,
                    eval_batch_size=256,
                    lr=0.02,
                    weight_decay=0,
                    max_steps=300000,
                    iter_per_step=1,
                    checkpoint_save_freq=10000,
                    eval_freq=2000)

parser.add_argument("--pretrained_model",
                    default="bert-base-uncased",
                    type=str)
parser.add_argument("--warmup_steps", default=4000, type=int)
parser.add_argument("--d_model", default=768, type=int)
parser.add_argument("--d_embedding", default=768, type=int)
parser.add_argument("--d_inner", default=3072, type=int)
parser.add_argument("--num_layers", default=12, type=int)
parser.add_argument("--num_heads", default=12, type=int)
parser.add_argument("--embedding_dropout", default=0.25, type=float)
parser.add_argument("--ffn_dropout", default=0.25, type=float)
parser.add_argument("--attn_score_dropout", default=0.25, type=float)
parser.add_argument("--attn_layer_dropout", default=0.25, type=float)
parser.add_argument("--eval_step_frequency", default=2000, type=int)
parser.add_argument("--dataset_dir", default="/dataset/", type=str)
parser.add_argument("--src_lang", default="pred", type=str)
parser.add_argument("--tgt_lang", default="real", type=str)
parser.add_argument("--beam_size", default=4, type=int)
parser.add_argument("--share_decoder_layers", action="store_true")
parser.add_argument("--restore_decoder", action="store_true")
parser.add_argument("--len_pen", default=0.0, type=float)
parser.add_argument("--fp16", default=1, type=int)
args = parser.parse_args()

# Start Tensorboard X for logging
tb_name = "asr_postprocessor-lr{0}-opt{1}-warmup{2}-{3}-bs{4}".format(
    args.lr, args.optimizer, args.warmup_steps, "poly", args.batch_size)

if args.fp16 == 3:
    opt_level = nemo.core.Optimization.mxprO3
elif args.fp16 == 2:
    opt_level = nemo.core.Optimization.mxprO2
elif args.fp16 == 1:
    opt_level = nemo.core.Optimization.mxprO1
else:
    opt_level = nemo.core.Optimization.mxprO0

neural_factory = nemo.core.NeuralModuleFactory(
    local_rank=args.local_rank,
    optimization_level=opt_level,
    log_dir=args.work_dir,
    create_tb_writer=False,
    tensorboard_dir=tb_name,
)

tokenizer = NemoBertTokenizer(pretrained_model=args.pretrained_model)

vocab_size = 8 * math.ceil(tokenizer.vocab_size / 8)

tokens_to_add = vocab_size - tokenizer.vocab_size
max_sequence_length = 512

train_data_layer = nemo_nlp.TranslationDataLayer(
    factory=neural_factory,
    tokenizer_src=tokenizer,
    tokenizer_tgt=tokenizer,
    dataset_src=args.dataset_dir + args.train_dataset + "." + args.src_lang,
    dataset_tgt=args.dataset_dir + args.train_dataset + "." + args.tgt_lang,
    tokens_in_batch=args.batch_size,
    clean=True)

eval_data_layers = {}

for key in args.eval_datasets:
    eval_data_layers[key] = nemo_nlp.TranslationDataLayer(
        factory=neural_factory,
        tokenizer_src=tokenizer,
        tokenizer_tgt=tokenizer,
        dataset_src=args.dataset_dir + key + "." + args.src_lang,
        dataset_tgt=args.dataset_dir + key + "." + args.tgt_lang,
        tokens_in_batch=args.eval_batch_size,
        clean=False)

do_lower_case = True

zeros_transform = nemo_nlp.ZerosLikeNM(factory=neural_factory)

encoder = nemo_nlp.huggingface.BERT(
    factory=neural_factory,
    pretrained_model_name=args.pretrained_model,
    local_rank=args.local_rank)

device = encoder.bert.embeddings.word_embeddings.weight.get_device()
zeros = torch.zeros((tokens_to_add, args.d_model)).to(device=device)

encoder.bert.embeddings.word_embeddings.weight.data = torch.cat(
    (encoder.bert.embeddings.word_embeddings.weight.data, zeros))

decoder = nemo_nlp.TransformerDecoderNM(
    factory=neural_factory,
    d_embedding=args.d_embedding,
    d_model=args.d_model,
    d_inner=args.d_inner,
    num_layers=args.num_layers,
    num_attn_heads=args.num_heads,
    ffn_dropout=args.ffn_dropout,
    vocab_size=vocab_size,
    attn_score_dropout=args.attn_score_dropout,
    attn_layer_dropout=args.attn_layer_dropout,
    max_seq_length=max_sequence_length,
    embedding_dropout=args.embedding_dropout,
    share_all_layers=args.share_decoder_layers,
    learn_positional_encodings=True,
    hidden_act="gelu")
if args.restore_decoder:
    decoder.restore_from(
        "../../scripts/bert-base-uncased_decoder.pt",
        local_rank=args.local_rank)

t_log_softmax = nemo_nlp.TransformerLogSoftmaxNM(
    factory=neural_factory,
    vocab_size=vocab_size,
    d_model=args.d_model,
    d_embedding=args.d_embedding)

beam_translator = nemo_nlp.BeamSearchTranslatorNM(
    factory=neural_factory,
    decoder=decoder,
    log_softmax=t_log_softmax,
    max_seq_length=max_sequence_length,
    beam_size=args.beam_size,
    length_penalty=args.len_pen,
    bos_token=tokenizer.bos_id(),
    pad_token=tokenizer.pad_id(),
    eos_token=tokenizer.eos_id())

loss = nemo_nlp.PaddedSmoothedCrossEntropyLossNM(factory=neural_factory,
                                                 pad_id=0,
                                                 smoothing=0.1)

loss_eval = nemo_nlp.PaddedSmoothedCrossEntropyLossNM(factory=neural_factory,
                                                      pad_id=0,
                                                      smoothing=0.0)

# tie all embeddings weights
t_log_softmax.log_softmax.dense.weight = \
    encoder.bert.embeddings.word_embeddings.weight
decoder.embedding_layer.token_embedding.weight = \
    encoder.bert.embeddings.word_embeddings.weight
decoder.embedding_layer.position_embedding.weight = \
    encoder.bert.embeddings.position_embeddings.weight

# training pipeline
src, src_mask, tgt, tgt_mask, labels, sent_ids = train_data_layer()

input_type_ids = zeros_transform(input_type_ids=src)
src_hiddens = encoder(input_ids=src,
                      token_type_ids=input_type_ids,
                      attention_mask=src_mask)
tgt_hiddens = decoder(input_ids_tgt=tgt,
                      hidden_states_src=src_hiddens,
                      input_mask_src=src_mask,
                      input_mask_tgt=tgt_mask)
log_softmax = t_log_softmax(hidden_states=tgt_hiddens)
train_loss = loss(log_probs=log_softmax, target_ids=labels)

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
log_softmax_ = {}
eval_loss_ = {}
beam_trans_ = {}

for key in args.eval_datasets:

    src_[key], src_mask_[key], tgt_[key], tgt_mask_[key], \
        labels_[key], sent_ids_[key] = eval_data_layers[key]()

    input_type_ids_[key] = zeros_transform(input_type_ids=src_[key])

    src_hiddens_[key] = encoder(input_ids=src_[key],
                                token_type_ids=input_type_ids_[key],
                                attention_mask=src_mask_[key])

    tgt_hiddens_[key] = decoder(input_ids_tgt=tgt_[key],
                                hidden_states_src=src_hiddens_[key],
                                input_mask_src=src_mask_[key],
                                input_mask_tgt=tgt_mask_[key])

    log_softmax_[key] = t_log_softmax(hidden_states=tgt_hiddens_[key])
    eval_loss_[key] = loss_eval(log_probs=log_softmax_[key],
                                target_ids=labels_[key])
    beam_trans_[key] = beam_translator(hidden_states_src=src_hiddens_[key],
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
        eval_step=args.eval_step_frequency,
        tb_writer=neural_factory.tb_writer)

    callbacks.append(callback)

checkpointer_callback = CheckpointCallback(folder=args.work_dir,
                                           step_freq=args.checkpoint_save_freq)
callbacks.append(checkpointer_callback)

# define learning rate decay policy
lr_policy = SquareAnnealing(total_steps=args.max_steps,
                            min_lr=1e-5,
                            warmup_steps=args.warmup_steps)

# Create trainer and execute training action
neural_factory.train(tensors_to_optimize=[train_loss],
                     callbacks=callbacks,
                     optimizer=args.optimizer,
                     lr_policy=lr_policy,
                     optimization_params={
                         "num_epochs": 300,
                         "lr": args.lr,
                         "weight_decay": args.weight_decay
                     },
                     batches_per_step=args.iter_per_step)
