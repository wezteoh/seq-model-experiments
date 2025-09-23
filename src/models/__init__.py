def get_model(name, args, device):
    if name == "mamba2":
        from .mamba2 import Mamba2Config, Mamba2MixerModel

        args = Mamba2Config(**args)
        return Mamba2MixerModel(args=args, device=device)
    elif name == "mamba2_fused":
        from .mamba2_fused import Mamba2MixerModel

        assert device == "cuda", "mamba2_fused only supports GPU"

        return Mamba2MixerModel(**args).to(device)

    elif name == "s4":
        from .s4 import S4Model

        return S4Model(**args)

    elif name == "transformer":
        from .transformer import TransformerModel

        return TransformerModel(**args)

    else:
        raise ValueError(f"Invalid model name: {name}")
