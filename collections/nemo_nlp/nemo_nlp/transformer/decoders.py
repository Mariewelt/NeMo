import copy
import torch
import torch.nn as nn
from .modules import MultiHeadAttention, PositionWiseFF
from .utils import form_attention_mask


class TransformerDecoderBlock(nn.Module):
    """
    Building block of Transformer decoder.

    Args:
        hidden_size: size of the embeddings in the model, also known as d_model
        inner_size: number of neurons in the intermediate part of feed-forward
            net, usually is (4-8 x hidden_size) in the papers
        num_attention_heads: number of heads in multi-head attention
        attn_score_dropout: probability of dropout applied to attention scores
        attn_layer_dropout: probability of dropout applied to the output of the
            attention layers, but before layer normalization
        ffn_dropout: probability of dropout applied to FFN output
        hidden_act: activation function used between two linear layers in FFN
    """

    def __init__(self, hidden_size, inner_size, num_attention_heads=1,
                 attn_score_dropout=0, attn_layer_dropout=0, ffn_dropout=0,
                 hidden_act="relu"):
        super().__init__()

        self.first_sub_layer = MultiHeadAttention(
            hidden_size, num_attention_heads,
            attn_score_dropout, attn_layer_dropout)
        self.second_sub_layer = MultiHeadAttention(
            hidden_size, num_attention_heads,
            attn_score_dropout, attn_layer_dropout)
        self.third_sub_layer = PositionWiseFF(
            hidden_size, inner_size, ffn_dropout, hidden_act)

    def forward(self, decoder_query, decoder_mask, decoder_keys,
                encoder_states, encoder_mask):
        self_attn_output = self.first_sub_layer(
            decoder_query, decoder_keys, decoder_keys, decoder_mask)
        enc_dec_attn_output = self.second_sub_layer(
            self_attn_output, encoder_states, encoder_states, encoder_mask)
        output_states = self.third_sub_layer(enc_dec_attn_output)
        return output_states


class TransformerDecoder(nn.Module):

    def __init__(self, num_layers, hidden_size, share_all_layers=False,
                 **kwargs):
        super().__init__()

        num_diff_layers = 1 if share_all_layers else num_layers
        layer = TransformerDecoderBlock(hidden_size, **kwargs)
        self.layers = nn.ModuleList(
            [copy.deepcopy(layer) for _ in range(num_diff_layers)])

        self.num_layers = num_layers
        self.share_all_layers = share_all_layers

    def _get_memory_states(self, decoder_states, decoder_mems_list=None, i=0):
        if decoder_mems_list is not None:
            memory_states = torch.cat(
                (decoder_mems_list[i], decoder_states), dim=1)
        else:
            memory_states = decoder_states
        return memory_states

    def forward(self, decoder_states, decoder_mask, encoder_states,
                encoder_mask, decoder_mems_list=None, return_mems=False):
        """
        Args:
            decoder_states: output of the embedding layer (B x L_dec x H)
            decoder_mask: decoder inputs mask (B x L_dec)
            encoder_states: output of the encoder (B x L_enc x H)
            encoder_mask: encoder inputs mask (B x L_enc)
            decoder_mems_list: list of the cached decoder hidden states
                for fast autoregressive generation which will be used instead
                of decoder_states as keys and values if not None
            return_mems: bool, whether to return outputs of all decoder layers
                or the last layer only
        """

        decoder_attn_mask = form_attention_mask(decoder_mask, diagonal=0)
        encoder_attn_mask = form_attention_mask(encoder_mask)

        memory_states = self._get_memory_states(
            decoder_states, decoder_mems_list, 0)
        cached_mems_list = [memory_states]

        for i in range(self.num_layers):
            layer_id = i * (1 - self.share_all_layers)
            decoder_states = self.layers[layer_id](
                decoder_states, decoder_attn_mask, memory_states,
                encoder_states, encoder_attn_mask)
            memory_states = self._get_memory_states(decoder_states,
                                                    decoder_mems_list, i + 1)
            cached_mems_list.append(memory_states)

        if return_mems:
            return cached_mems_list
        else:
            return cached_mems_list[-1]
