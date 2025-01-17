# Copyright (c) 2019 NVIDIA Corporation
import math

import nemo
from nemo.utils.lr_policies import CosineAnnealing
import nemo_nlp
from nemo_nlp.callbacks.language_modeling import eval_iter_callback, \
    eval_epochs_done_callback
from nemo_nlp.text_data_utils import LanguageModelDataDesc


parser = nemo.utils.NemoArgParser(description='LM Transformer')
parser.set_defaults(
    train_dataset="train.txt",
    eval_datasets=["valid.txt"],
    work_dir="outputs/transformer_lm",
    optimizer_kind="novograd",
    amp_opt_level='O2',
    num_epochs=1000,
    batch_size=32,
    eval_batch_size=32,
    lr=0.002,
    beta1=0.95,
    beta2=0.25,
    weight_decay=0,
    warmup_steps=1000,
    max_steps=50000,
    iter_per_step=1,
    checkpoint_save_freq=10000,
    eval_freq=1000
)
parser.add_argument("--data_dir", default="data/lm/wikitext-2", type=str)
parser.add_argument("--dataset_name", default="wikitext-2", type=str)
parser.add_argument("--d_model", default=384, type=int)
parser.add_argument("--d_inner", default=1536, type=int)
parser.add_argument("--num_layers", default=12, type=int)
parser.add_argument("--num_attn_heads", default=6, type=int)
parser.add_argument("--embedding_dropout", default=0.2, type=float)
parser.add_argument("--ffn_dropout", default=0.2, type=float)
parser.add_argument("--attn_score_dropout", default=0.2, type=float)
parser.add_argument("--attn_layer_dropout", default=0.2, type=float)
parser.add_argument("--max_sequence_length", default=256, type=int)
parser.add_argument("--do_lower_case", action='store_true')
parser.add_argument("--label_smoothing", default=0.1, type=float)
parser.add_argument("--beam_size", default=4, type=int)
parser.add_argument("--tokenizer_model", default="vocab.txt", type=str)
parser.add_argument("--predict_last_k", default=16, type=int)
parser.add_argument("--interactive", action="store_true")
args = parser.parse_args()

"""
To get the data, go to tests/data and run get_wt2.sh
Then run create_vocab.py
"""

nf = nemo.core.NeuralModuleFactory(backend=nemo.core.Backend.PyTorch,
                                   local_rank=args.local_rank,
                                   optimization_level=args.amp_opt_level,
                                   log_dir=args.work_dir,
                                   create_tb_writer=True,
                                   files_to_copy=[__file__])

data_desc = LanguageModelDataDesc(
    args.dataset_name, args.data_dir, args.do_lower_case)

# define tokenizer, in this example we use word-level tokenizer
# we also adjust the vocabulary size to make it multiple of 8 to accelerate
# training in fp16 mode with the use of Tensor Cores
tokenizer = nemo_nlp.WordTokenizer(f"{args.data_dir}/{args.tokenizer_model}")
vocab_size = 8 * math.ceil(tokenizer.vocab_size / 8)

# instantiate necessary modules for the whole translation pipeline, namely
# data layers, encoder, decoder, output log_softmax, beam_search_translator
# and loss function


train_dataset = nemo_nlp.LanguageModelingDataset(
    tokenizer,
    dataset=f"{args.data_dir}/{args.train_dataset}",
    max_sequence_length=args.max_sequence_length,
    batch_step=args.max_sequence_length)

eval_dataset = nemo_nlp.LanguageModelingDataset(
    tokenizer,
    dataset=f"{args.data_dir}/{args.eval_datasets[0]}",
    max_sequence_length=args.max_sequence_length,
    batch_step=args.predict_last_k)

encoder = nemo_nlp.TransformerEncoderNM(
    d_model=args.d_model,
    d_inner=args.d_inner,
    num_layers=args.num_layers,
    embedding_dropout=args.embedding_dropout,
    num_attn_heads=args.num_attn_heads,
    ffn_dropout=args.ffn_dropout,
    vocab_size=vocab_size,
    mask_future=True,
    attn_score_dropout=args.attn_score_dropout,
    attn_layer_dropout=args.attn_layer_dropout,
    max_seq_length=args.max_sequence_length)

log_softmax = nemo_nlp.TransformerLogSoftmaxNM(
    vocab_size=vocab_size,
    d_model=args.d_model)

loss = nemo_nlp.PaddedSmoothedCrossEntropyLossNM(
    pad_id=tokenizer.pad_id(),
    label_smoothing=args.label_smoothing)

# tie weight of embedding and log_softmax layers
log_softmax.log_softmax.dense.weight = \
    encoder.embedding_layer.token_embedding.weight


def create_pipeline(dataset, batch_size):
    data_layer = nemo_nlp.LanguageModelingDataLayer(dataset,
                                                    batch_size=batch_size)
    src, src_mask, labels = data_layer()
    src_hiddens = encoder(input_ids=src, input_mask_src=src_mask)
    log_probs = log_softmax(hidden_states=src_hiddens)
    return loss(log_probs=log_probs, target_ids=labels)


train_loss = create_pipeline(train_dataset, args.batch_size)
eval_loss = create_pipeline(eval_dataset, args.batch_size)

# callback which prints training loss once in a while
callback_train = nemo.core.SimpleLossLoggerCallback(
    tensors=[train_loss],
    step_freq=100,
    print_func=lambda x: str(x[0].item()),
    get_tb_values=lambda x: [["loss", x[0]]],
    tb_writer=nf.tb_writer)

# callback which calculates evaluation loss
callback_eval = nemo.core.EvaluatorCallback(
    eval_tensors=[eval_loss],
    user_iter_callback=eval_iter_callback,
    user_epochs_done_callback=eval_epochs_done_callback,
    eval_step=args.eval_freq,
    tb_writer=nf.tb_writer)

# callback which saves checkpoints once in a while
callback_ckpt = nemo.core.CheckpointCallback(
    folder=nf.checkpoint_dir,
    step_freq=args.checkpoint_save_freq,
    checkpoints_to_keep=-1)

# define learning rate decay policy
lr_policy_fn = CosineAnnealing(args.max_steps, warmup_steps=args.warmup_steps)

# define and launch training algorithm (optimizer)
max_num_epochs = 0 if args.interactive else args.num_epochs

callbacks = [callback_ckpt]

if not args.interactive:
    callbacks.extend([callback_train, callback_eval])

nf.train(tensors_to_optimize=[train_loss],
         callbacks=callbacks,
         lr_policy=lr_policy_fn,
         batches_per_step=args.iter_per_step,
         optimizer=args.optimizer_kind,
         optimization_params={"num_epochs": args.num_epochs,
                              "lr": args.lr,
                              "weight_decay": args.weight_decay,
                              "betas": (args.beta1, args.beta2)})
