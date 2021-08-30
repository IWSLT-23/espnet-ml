"""Transducer speech recognition model (pytorch)."""

from argparse import Namespace
from collections import Counter
from dataclasses import asdict
import logging
import math
import numpy

import chainer
import torch

from espnet.nets.asr_interface import ASRInterface
from espnet.nets.pytorch_backend.ctc import ctc_for
from espnet.nets.pytorch_backend.nets_utils import get_subsample
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask
from espnet.nets.pytorch_backend.transducer.arguments import (
    add_encoder_general_arguments,  # noqa: H301
    add_rnn_encoder_arguments,  # noqa: H301
    add_custom_encoder_arguments,  # noqa: H301
    add_decoder_general_arguments,  # noqa: H301
    add_rnn_decoder_arguments,  # noqa: H301
    add_custom_decoder_arguments,  # noqa: H301
    add_custom_training_arguments,  # noqa: H301
    add_transducer_arguments,  # noqa: H301
    add_auxiliary_task_arguments,  # noqa: H301
)
from espnet.nets.pytorch_backend.transducer.auxiliary_task import AuxiliaryTask
from espnet.nets.pytorch_backend.transducer.custom_decoder import CustomDecoder
from espnet.nets.pytorch_backend.transducer.custom_encoder import CustomEncoder
from espnet.nets.pytorch_backend.transducer.error_calculator import ErrorCalculator
from espnet.nets.e2e_asr_common import ErrorCalculator as ErrorCalculatorCTC
from espnet.nets.pytorch_backend.transducer.initializer import initializer
from espnet.nets.pytorch_backend.transducer.joint_network import JointNetwork
from espnet.nets.pytorch_backend.transducer.loss import TransLoss
from espnet.nets.pytorch_backend.transducer.rnn_decoder import DecoderRNNT
from espnet.nets.pytorch_backend.transducer.rnn_encoder import encoder_for
from espnet.nets.pytorch_backend.transducer.utils import prepare_loss_inputs
from espnet.nets.pytorch_backend.transducer.utils import valid_aux_task_layer_list
from espnet.nets.pytorch_backend.transformer.attention import (
    MultiHeadedAttention,  # noqa: H301
    RelPositionMultiHeadedAttention,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import (
    LabelSmoothingLoss,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.mask import target_mask
from espnet.nets.pytorch_backend.transformer.plot import PlotAttentionReport
from espnet.utils.fill_missing_args import fill_missing_args

from espnet.nets.pytorch_backend.conformer.argument import (
    add_arguments_conformer_common,  # noqa: H301
    verify_rel_pos_type,  # noqa: H301
)

from espnet.nets.pytorch_backend.transformer.argument import (
    add_arguments_transformer_for_cond1,  # noqa: H301
)

from espnet.nets.pytorch_backend.conformer.encoder import Encoder
from espnet.nets.pytorch_backend.transformer.encoder import Encoder as Encoder_T
from espnet.nets.pytorch_backend.ctc import CTC

class Reporter(chainer.Chain):
    """A chainer reporter wrapper for transducer models."""

    def report(
        self,
        loss,
        loss_trans,
        loss_ctc,
        loss_lm,
        loss_aux_trans,
        loss_aux_symm_kl,
        cer,
        wer,
        zh_ctc_loss,
        en_ctc_loss,
        zh_cer,
        en_cer
    ):
        """Instantiate reporter attributes."""
        chainer.reporter.report({"loss": loss}, self)
        chainer.reporter.report({"loss_trans": loss_trans}, self)
        chainer.reporter.report({"loss_ctc": loss_ctc}, self)
        chainer.reporter.report({"loss_lm": loss_lm}, self)
        chainer.reporter.report({"loss_aux_trans": loss_aux_trans}, self)
        chainer.reporter.report({"loss_aux_symm_kl": loss_aux_symm_kl}, self)
        chainer.reporter.report({"cer": cer}, self)
        chainer.reporter.report({"wer": wer}, self)
        chainer.reporter.report({"zh_ctc_loss": zh_ctc_loss}, self)
        chainer.reporter.report({"en_ctc_loss": en_ctc_loss}, self)
        chainer.reporter.report({"zh_cer": zh_cer}, self)
        chainer.reporter.report({"en_cer": en_cer}, self)

        logging.info("loss:" + str(loss))


class E2E(ASRInterface, torch.nn.Module):
    """E2E module for transducer models.

    Args:
        idim (int): dimension of inputs
        odim (int): dimension of outputs
        args (Namespace): argument Namespace containing options
        ignore_id (int): padding symbol id
        blank_id (int): blank symbol id

    """

    @staticmethod
    def add_arguments(parser):
        """Add arguments for transducer model."""
        E2E.encoder_add_general_arguments(parser)
        E2E.encoder_add_rnn_arguments(parser)
        E2E.encoder_add_custom_arguments(parser)

        E2E.decoder_add_general_arguments(parser)
        E2E.decoder_add_rnn_arguments(parser)
        E2E.decoder_add_custom_arguments(parser)

        E2E.training_add_custom_arguments(parser)
        E2E.transducer_add_arguments(parser)
        E2E.auxiliary_task_add_arguments(parser)

        group = parser.add_argument_group("conformer model specific setting")
        group = add_arguments_conformer_common(group)

        group = parser.add_argument_group("transformer model setting")
        group = add_arguments_transformer_for_cond1(group)
        return parser

    @staticmethod
    def encoder_add_general_arguments(parser):
        """Add general arguments for encoder."""
        group = parser.add_argument_group("Encoder general arguments")
        group = add_encoder_general_arguments(group)

        return parser

    @staticmethod
    def encoder_add_rnn_arguments(parser):
        """Add arguments for RNN encoder."""
        group = parser.add_argument_group("RNN encoder arguments")
        group = add_rnn_encoder_arguments(group)

        return parser

    @staticmethod
    def encoder_add_custom_arguments(parser):
        """Add arguments for Custom encoder."""
        group = parser.add_argument_group("Custom encoder arguments")
        group = add_custom_encoder_arguments(group)

        return parser

    @staticmethod
    def decoder_add_general_arguments(parser):
        """Add general arguments for decoder."""
        group = parser.add_argument_group("Decoder general arguments")
        group = add_decoder_general_arguments(group)

        return parser

    @staticmethod
    def decoder_add_rnn_arguments(parser):
        """Add arguments for RNN decoder."""
        group = parser.add_argument_group("RNN decoder arguments")
        group = add_rnn_decoder_arguments(group)

        return parser

    @staticmethod
    def decoder_add_custom_arguments(parser):
        """Add arguments for Custom decoder."""
        group = parser.add_argument_group("Custom decoder arguments")
        group = add_custom_decoder_arguments(group)

        return parser

    @staticmethod
    def training_add_custom_arguments(parser):
        """Add arguments for Custom architecture training."""
        group = parser.add_argument_group("Training arguments for custom archictecture")
        group = add_custom_training_arguments(group)

        return parser

    @staticmethod
    def transducer_add_arguments(parser):
        """Add arguments for transducer model."""
        group = parser.add_argument_group("Transducer model arguments")
        group = add_transducer_arguments(group)

        return parser

    @staticmethod
    def auxiliary_task_add_arguments(parser):
        """Add arguments for auxiliary task."""
        group = parser.add_argument_group("Auxiliary task arguments")
        group = add_auxiliary_task_arguments(group)

        return parser

    @property
    def attention_plot_class(self):
        """Get attention plot class."""
        return PlotAttentionReport

    def get_total_subsampling_factor(self):
        """Get total subsampling factor."""
        # if self.etype == "custom":
        #     return self.encoder.conv_subsampling_factor * int(
        #         numpy.prod(self.subsample)
        #     )
        # else:
        #     return self.enc.conv_subsampling_factor * int(numpy.prod(self.subsample))
        return self.zh_encoder.conv_subsampling_factor * int(numpy.prod(self.subsample))

    def __init__(self, idim, odim, args, ignore_id=-1, blank_id=0, training=True):
        """Construct an E2E object for transducer model."""
        torch.nn.Module.__init__(self)

        args = fill_missing_args(args, self.add_arguments)

        self.is_rnnt = True
        self.transducer_weight = args.transducer_weight

        self.use_aux_task = (
            True if (args.aux_task_type is not None and training) else False
        )

        self.use_aux_ctc = args.aux_ctc and training
        self.aux_ctc_weight = args.aux_ctc_weight

        self.use_aux_cross_entropy = args.aux_cross_entropy and training
        self.aux_cross_entropy_weight = args.aux_cross_entropy_weight

        if self.use_aux_task:
            n_layers = (
                (len(args.enc_block_arch) * args.enc_block_repeat - 1)
                if args.enc_block_arch is not None
                else (args.elayers - 1)
            )

            aux_task_layer_list = valid_aux_task_layer_list(
                args.aux_task_layer_list,
                n_layers,
            )
        else:
            aux_task_layer_list = []

        self.subsample = get_subsample(args, mode="asr", arch="transformer")
        self.most_dom_list = args.enc_block_arch[:]

        if "custom" in args.dtype:
            if args.dec_block_arch is None:
                raise ValueError(
                    "When specifying custom decoder type, --dec-block-arch"
                    "should also be specified in training config. See"
                    "egs/vivos/asr1/conf/transducer/train_*.yaml for more info."
                )

            self.decoder = CustomDecoder(
                odim,
                args.dec_block_arch,
                input_layer=args.custom_dec_input_layer,
                repeat_block=args.dec_block_repeat,
                positionwise_activation_type=args.custom_dec_pw_activation_type,
                dropout_rate_embed=args.dropout_rate_embed_decoder,
            )
            decoder_out = self.decoder.dunits

            if "custom" in args.etype:
                self.most_dom_list += args.dec_block_arch[:]
            else:
                self.most_dom_list = args.dec_block_arch[:]
        else:
            self.dec = DecoderRNNT(
                odim,
                args.dtype,
                args.dlayers,
                args.dunits,
                blank_id,
                args.dec_embed_dim,
                args.dropout_rate_decoder,
                args.dropout_rate_embed_decoder,
            )
            decoder_out = args.dunits

        self.joint_network = JointNetwork(
            odim, args.adim, decoder_out, args.joint_dim, args.joint_activation_type
        )

        if hasattr(self, "most_dom_list"):
            self.most_dom_dim = sorted(
                Counter(
                    d["d_hidden"] for d in self.most_dom_list if "d_hidden" in d
                ).most_common(),
                key=lambda x: x[0],
                reverse=True,
            )[0][0]

        self.etype = args.etype
        self.dtype = args.dtype

        self.sos = odim - 1
        self.eos = odim - 1
        self.blank_id = blank_id
        self.ignore_id = ignore_id

        self.space = args.sym_space
        self.blank = args.sym_blank

        self.odim = odim

        self.reporter = Reporter()

        self.error_calculator = None

        self.default_parameters(args)

        if training:
            logging.warning("Using transducer package:" + str(args.trans_type))
            self.criterion = TransLoss(args.trans_type, self.blank_id)

            decoder = self.decoder if self.dtype == "custom" else self.dec

            if args.report_cer or args.report_wer:
                self.error_calculator = ErrorCalculator(
                    decoder,
                    self.joint_network,
                    args.char_list,
                    args.sym_space,
                    args.sym_blank,
                    args.report_cer,
                    args.report_wer,
                )
                self.error_calculator_ctc = ErrorCalculatorCTC(
                    args.char_list,
                    args.sym_space,
                    args.sym_blank,
                    args.report_cer,
                    args.report_wer,
                )

            if self.use_aux_task:
                self.auxiliary_task = AuxiliaryTask(
                    decoder,
                    self.joint_network,
                    self.criterion,
                    args.aux_task_type,
                    args.aux_task_weight,
                    encoder_out,
                    args.joint_dim,
                )

            if self.use_aux_ctc:
                self.aux_ctc = ctc_for(
                    Namespace(
                        num_encs=1,
                        eprojs=encoder_out,
                        dropout_rate=args.aux_ctc_dropout_rate,
                        ctc_type="warpctc",
                    ),
                    odim,
                )

            if self.use_aux_cross_entropy:
                self.aux_decoder_output = torch.nn.Linear(decoder_out, odim)

                self.aux_cross_entropy = LabelSmoothingLoss(
                    odim, ignore_id, args.aux_cross_entropy_smoothing
                )

        self.loss = None
        self.rnnlm = None

        # Cond
        self.cond_weight = args.cond_weight
        self.fusion_type = args.fusion_type
        if hasattr(args, 'gt_mask_type'):
            self.gt_mask_type = args.gt_mask_type
        else:
            self.gt_mask_type = "ignore"
        # zh encoder
        if args.transformer_attn_dropout_rate is None:
            args.transformer_attn_dropout_rate = args.dropout_rate

        logging.warning("attn dropout:" + str(args.transformer_attn_dropout_rate))
        self.cond_adim = args.adim

        self.encoder = Encoder(
            idim=idim,
            attention_dim=args.adim,
            attention_heads=args.aheads,
            linear_units=args.cond_eunits,
            num_blocks=args.cond_elayers,
            input_layer=args.transformer_input_layer,
            dropout_rate=args.cond_dropout_rate,
            positional_dropout_rate=args.cond_dropout_rate,
            attention_dropout_rate=args.transformer_attn_dropout_rate,
            pos_enc_layer_type=args.transformer_encoder_pos_enc_layer_type,
            selfattention_layer_type=args.transformer_encoder_selfattn_layer_type,
            activation_type=args.transformer_encoder_activation_type,
            macaron_style=args.macaron_style,
            use_cnn_module=args.use_cnn_module,
            zero_triu=args.zero_triu,
            cnn_module_kernel=args.cnn_module_kernel,
        )

        if not hasattr(args, 'lang_elayers'):
            args.lang_elayers = 2
        self.zh_encoder = Encoder_T(
            idim=odim,
            attention_dim=args.adim,
            attention_heads=args.aheads,
            linear_units=args.cond_eunits,
            num_blocks=args.lang_elayers,
            input_layer="nothing",
            dropout_rate=args.cond_dropout_rate,
            positional_dropout_rate=args.cond_dropout_rate,
            attention_dropout_rate=args.transformer_attn_dropout_rate,
            padding_idx=0,
        )
        self.zh_ctc = CTC(
            odim, args.adim, args.cond_dropout_rate, ctc_type=args.ctc_type, reduce=True
        )
        # en encoder
        self.en_encoder = Encoder_T(
            idim=odim,
            attention_dim=args.adim,
            attention_heads=args.aheads,
            linear_units=args.cond_eunits,
            num_blocks=args.lang_elayers,
            input_layer="nothing",
            dropout_rate=args.cond_dropout_rate,
            positional_dropout_rate=args.cond_dropout_rate,
            attention_dropout_rate=args.transformer_attn_dropout_rate,
            padding_idx=0,
        )
        self.en_ctc = CTC(
            odim, args.adim, args.cond_dropout_rate, ctc_type=args.ctc_type, reduce=True
        )

    def default_parameters(self, args):
        """Initialize/reset parameters for transducer.

        Args:
            args (Namespace): argument Namespace containing options

        """
        initializer(self, args)

    def forward(self, xs_pad, ilens, ys_pad):
        """E2E forward.

        Args:
            xs_pad (torch.Tensor): batch of padded source sequences (B, Tmax, idim)
            ilens (torch.Tensor): batch of lengths of input sequences (B)
            ys_pad (torch.Tensor): batch of padded target sequences (B, Lmax)

        Returns:
            loss (torch.Tensor): transducer loss value

        """
        
        # Cond encoders
        xs_pad = xs_pad[:, : max(ilens)]
        src_mask = make_non_pad_mask(ilens.tolist()).to(xs_pad.device).unsqueeze(-2)
        # shared layers
        hs_pad, hs_mask = self.encoder(xs_pad, src_mask)

        # lang specific layers
        zh_hs_pad, _ = self.zh_encoder(hs_pad, hs_mask)
        en_hs_pad, _ = self.en_encoder(hs_pad, hs_mask)
        # make zh and en gts
        if self.gt_mask_type == "lid":
            # zh is all from 5000 onwards
            zh_ys_pad = ys_pad.masked_fill(~(ys_pad >= 5000), 11238)
            # en is < 5000 and including lid tags
            en_ys_pad = ys_pad.masked_fill(~((ys_pad < 5000) | (ys_pad >= 11237)), 11237)
        else:
            # zh is all from 5000 onwards
            zh_ys_pad = ys_pad.masked_fill(~(ys_pad >= 5000), -1)
            # en is < 5000 and including lid tags
            en_ys_pad = ys_pad.masked_fill(~((ys_pad < 5000) | (ys_pad >= 11237)), -1)
        # ctc losses
        batch_size = xs_pad.size(0)
        hs_len = hs_mask.view(batch_size, -1).sum(1)
        if self.fusion_type == "confidence":
            zh_ctc_loss, zh_posteriors = self.zh_ctc(zh_hs_pad.view(batch_size, -1, self.cond_adim), hs_len, zh_ys_pad, get_posteriors=True)
            en_ctc_loss, en_posteriors = self.en_ctc(en_hs_pad.view(batch_size, -1, self.cond_adim), hs_len, en_ys_pad, get_posteriors=True)
        else:
            zh_ctc_loss = self.zh_ctc(zh_hs_pad.view(batch_size, -1, self.cond_adim), hs_len, zh_ys_pad)
            en_ctc_loss = self.en_ctc(en_hs_pad.view(batch_size, -1, self.cond_adim), hs_len, en_ys_pad)
        # logging
        if not self.training and self.error_calculator_ctc is not None:
            zh_ys_hat = self.zh_ctc.argmax(zh_hs_pad.view(batch_size, -1, self.cond_adim)).data
            zh_cer_ctc = self.error_calculator_ctc(zh_ys_hat.cpu(), zh_ys_pad.cpu(), is_ctc=True)

            en_ys_hat = self.en_ctc.argmax(en_hs_pad.view(batch_size, -1, self.cond_adim)).data
            en_cer_ctc = self.error_calculator_ctc(en_ys_hat.cpu(), en_ys_pad.cpu(), is_ctc=True)
        else:
            zh_cer_ctc = None
            en_cer_ctc = None

        # 1.5. transducer preparation related
        ys_in_pad, ys_out_pad, target, pred_len, target_len = prepare_loss_inputs(
            ys_pad, hs_mask
        )

        # Fusion of Cond encoders with rnnt encoder
        if self.fusion_type == "add":
            # hs_pad = hs_pad + zh_hs_pad + en_hs_pad
            hs_pad = zh_hs_pad + en_hs_pad
        elif self.fusion_type == "confidence":
            # zh_ys_hat is B x T x V --> confidence is B x T x 1
            # zh_hs_pad is B x T x D
            zh_confidence = zh_posteriors[:,:,-3]
            zh_hs_pad = zh_hs_pad * zh_confidence.unsqueeze(-1)
            en_confidence = en_posteriors[:,:,-2]
            en_hs_pad = en_hs_pad * en_confidence.unsqueeze(-1)
            hs_pad = zh_hs_pad + en_hs_pad
        else:
            import pdb;pdb.set_trace()

        # 2. decoder
        if "custom" in self.dtype:
            ys_mask = target_mask(ys_in_pad, self.blank_id)
            pred_pad, _ = self.decoder(ys_in_pad, ys_mask, hs_pad)
        else:
            pred_pad = self.dec(hs_pad, ys_in_pad)

        z = self.joint_network(hs_pad.unsqueeze(2), pred_pad.unsqueeze(1))

        # 3. loss computation
        loss_trans = self.criterion(z, target, pred_len, target_len)

        loss = (
            ((1 - self.cond_weight) * loss_trans)
            + ((self.cond_weight / 2) * zh_ctc_loss)
            + ((self.cond_weight / 2) * en_ctc_loss)
        )
        self.loss = loss
        loss_data = float(loss)

        # 4. compute cer/wer
        if self.training or self.error_calculator is None:
            cer, wer = None, None
        else:
            cer, wer = self.error_calculator(hs_pad.cpu(), ys_pad.cpu())

        if not math.isnan(loss_data):
            self.reporter.report(
                loss_data,
                float(loss_trans),
                None, None, None, None,
                # float(loss_ctc),
                # float(loss_lm),
                # float(loss_aux_trans),
                # float(loss_aux_symm_kl),
                cer,
                wer,
                float(zh_ctc_loss),
                float(en_ctc_loss),
                zh_cer_ctc,
                en_cer_ctc,
            )
        else:
            logging.warning("loss (=%f) is not correct", loss_data)

        return self.loss

    def encode_custom(self, x):
        """Encode acoustic features.

        Args:
            x (ndarray): input acoustic feature (T, D)

        Returns:
            x (torch.Tensor): encoded features (T, D_enc)

        """
        x = torch.as_tensor(x).unsqueeze(0)
        enc_output, _ = self.encoder(x, None)

        return enc_output.squeeze(0)

    def encode_rnn(self, x):
        """Encode acoustic features.

        Args:
            x (ndarray): input acoustic feature (T, D)

        Returns:
            x (torch.Tensor): encoded features (T, D_enc)

        """
        p = next(self.parameters())

        ilens = [x.shape[0]]
        x = x[:: self.subsample[0], :]

        h = torch.as_tensor(x, device=p.device, dtype=p.dtype)
        hs = h.contiguous().unsqueeze(0)

        hs, _, _ = self.enc(hs, ilens)

        return hs.squeeze(0)

    def recognize(self, x, beam_search):
        """Recognize input features.

        Args:
            x (ndarray): input acoustic feature (T, D)
            beam_search (class): beam search class

        Returns:
            nbest_hyps (list): n-best decoding results

        """
        self.eval()
        x = torch.as_tensor(x).unsqueeze(0)
        h, _ = self.encoder(x, None)
        zh_h, _ = self.zh_encoder(h, None)
        en_h, _ = self.en_encoder(h, None)

        # # # tmp code to try to view ctc outputs
        # from itertools import groupby
        # lpz = self.zh_ctc.argmax(zh_h)
        # # lpz = self.en_ctc.argmax(en_h)
        # collapsed_indices = [x[0] for x in groupby(lpz[0])]
        # hyp = [x for x in filter(lambda x: x != self.blank, collapsed_indices)]
        # nbest_hyps = [{"score": 0.0, "yseq": [self.sos] + hyp}]
        # return nbest_hyps

        # Fusion of Cond encoders with rnnt encoder
        if self.fusion_type == "add":
            # h = h + zh_h + en_h
            h = zh_h + en_h
        else:
            import pdb;pdb.set_trace()

        h = h.squeeze(0)

        nbest_hyps = beam_search(h)

        return [asdict(n) for n in nbest_hyps]

    def calculate_all_attentions(self, xs_pad, ilens, ys_pad):
        """E2E attention calculation.

        Args:
            xs_pad (torch.Tensor): batch of padded input sequences (B, Tmax, idim)
            ilens (torch.Tensor): batch of lengths of input sequences (B)
            ys_pad (torch.Tensor):
                batch of padded character id sequence tensor (B, Lmax)

        Returns:
            ret (ndarray): attention weights with the following shape,
                1) multi-head case => attention weights (B, H, Lmax, Tmax),
                2) other case => attention weights (B, Lmax, Tmax).

        """
        self.eval()

        if "custom" not in self.etype and "custom" not in self.dtype:
            return []
        else:
            with torch.no_grad():
                self.forward(xs_pad, ilens, ys_pad)

            ret = dict()
            for name, m in self.named_modules():
                if isinstance(m, MultiHeadedAttention) or isinstance(
                    m, RelPositionMultiHeadedAttention
                ):
                    if m.attn is None:
                        continue
                    ret[name] = m.attn.cpu().numpy()

        self.train()

        return ret
