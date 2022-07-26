from contextlib import contextmanager
from distutils.version import LooseVersion
from collections import defaultdict
import argparse
import torch
import sys
import numpy as np
from tqdm import tqdm

from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention

# from espnet2.tasks.st import STTask
from espnet2.tasks.asr import ASRTask
from espnet2.utils.types import str2bool
from espnet2.utils.types import str2triple_str
from espnet2.utils.types import str_or_none
from espnet2.torch_utils.device_funcs import to_device
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos



if LooseVersion(torch.__version__) >= LooseVersion("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield




def get_parser():
    parser = argparse.ArgumentParser(
        description="Calculate the diagonality of the self-attention weights."
    )
    parser.add_argument(
        "--asr_train_config",
        type=str,
        help="path to the asr train config file"
    )
    parser.add_argument(
        "--asr_model_file",
        type=str,
        help="path to the trained model file"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=['cpu', 'gpu'],
        default='cpu',
        help="device name: cpu (default), gpu"
    )
    parser.add_argument(
        "--dst",
        type=str,
        help="log file name"
    )
    parser.add_argument(
        "--data_path_and_name_and_type",
        type=str2triple_str,
        required=True,
        action="append",
    )
    parser.add_argument(
        "--key_file", 
        type=str_or_none,
        help="wav.scp"
    )
    parser.add_argument(
        "--allow_variable_data_keys", 
        type=str2bool, 
        default=False
    )
    # parser.add_argument(
    #     "--use_hier_ctc", 
    #     type=str2bool, 
    #     default=True
    # )
    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()


    if args.device == 'gpu':
        args.device = 'cuda'

    asr_model, asr_train_args = ASRTask.build_model_from_file(
        args.asr_train_config, args.asr_model_file, args.device
    )
    asr_model.eval()

    # dataloader
    loader = ASRTask.build_streaming_iterator(
        args.data_path_and_name_and_type,
        dtype="float32",
        batch_size=1,       # mush be 1, otherwise there will be paddings
        key_file=args.key_file,
        num_workers=2,
        preprocess_fn=ASRTask.build_preprocess_fn(asr_train_args, False),
        collate_fn=ASRTask.build_collate_fn(asr_train_args, False),
        allow_variable_data_keys=args.allow_variable_data_keys,
        inference=False,
    )

    # import pdb;pdb.set_trace()
    i = 0
    for keys, batch in tqdm(loader):
        # assert isinstance(batch, dict), type(batch)
        # assert all(isinstance(s, str) for s in keys), keys
        # _bs = len(next(iter(batch.values())))
        # assert len(keys) == _bs, f"{len(keys)} != {_bs}"
        # batch = {k: v[0] for k, v in batch.items() if not k.endswith("_lengths")}
        import pdb;pdb.set_trace()


        speech = batch["speech"].to(getattr(torch, "float32"))  # (B, T)
        lengths = batch["speech_lengths"]
        # text = batch["text"]
        # text_lengths = batch["text_lengths"]
        batch = {"speech": speech, "speech_lengths": lengths}

        # a. To device
        batch = to_device(batch, device=args.device)

        # b. Forward Encoder
        enc, enc_en, enc_zh = asr_model.encode_joint(**batch)

        bi_post = asr_model.ctc.softmax(enc).detach().numpy()
        en_post = asr_model.en_ctc.softmax(enc_en).detach().numpy()
        zh_post = asr_model.zh_ctc.softmax(enc_zh).detach().numpy()

        bi_post.dump(args.dst+"/bi"+str(i)+".np")
        en_post.dump(args.dst+"/en"+str(i)+".np")
        zh_post.dump(args.dst+"/zh"+str(i)+".np")

        i+=1
        if i > 100:
            break
        # import pdb;pdb.set_trace()