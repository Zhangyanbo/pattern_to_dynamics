# -*- coding: utf-8 -*-
from typing import Any, Dict
import torch.nn as nn
from diffusers import UNet2DModel


class UNet2DModelWithPadding(UNet2DModel):
    """
    Drop-in subclass that adds `padding_mode` as a first-class init parameter.
    The value is persisted in `config.json` and automatically restored via
    `from_pretrained` (through `from_config`) without needing to pass it again.
    """

    def __init__(
        self,
        *args,
        padding_mode: str = "zeros",
        only_when_effective: bool = False,
        log_changed: bool = False,
        **kwargs,
    ):
        """
        Args:
            padding_mode: one of {'zeros', 'reflect', 'replicate', 'circular'}.
            only_when_effective: if True, only change convs where (padding>0 and kernel_size>1).
            log_changed: if True, print which conv layers were modified.
        """
        super().__init__(*args, **kwargs)
        # Persist new fields so save_pretrained() writes them into config.json
        self.register_to_config(
            padding_mode=padding_mode,
            only_when_effective=only_when_effective,
        )
        self._apply_padding_mode(verbose=log_changed)

    @classmethod
    def from_config(cls, config: Dict[str, Any], **kwargs):
        """
        Ensure our extra config fields flow back into __init__ when loading.
        """
        # config may be an immutable mapping; make a shallow copy
        cfg = dict(config)

        # Read our fields from config (fallbacks keep backward compatibility)
        pm = cfg.pop("padding_mode", "zeros")
        owe = cfg.pop("only_when_effective", False)

        # Allow explicit runtime overrides (rare, but useful)
        pm = kwargs.pop("padding_mode", pm)
        owe = kwargs.pop("only_when_effective", owe)
        log = kwargs.pop("log_changed", False)

        # Pass the rest of original UNet2DModel kwargs through
        return cls(
            *(), padding_mode=pm, only_when_effective=owe, log_changed=log, **cfg
        )

    def _apply_padding_mode(self, *, verbose: bool = False):
        pm = getattr(self.config, "padding_mode", "zeros")
        only = getattr(self.config, "only_when_effective", False)

        modified = []
        unchanged = 0

        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                ks = (
                    m.kernel_size
                    if isinstance(m.kernel_size, tuple)
                    else (m.kernel_size, m.kernel_size)
                )
                pd = (
                    m.padding
                    if isinstance(m.padding, tuple)
                    else (m.padding, m.padding)
                )
                effective = (max(pd) > 0 and max(ks) > 1) if only else True

                if effective and m.padding_mode != pm:
                    old = m.padding_mode
                    m.padding_mode = pm
                    if verbose:
                        modified.append((name, old, m.kernel_size, m.padding))
                else:
                    unchanged += 1

        if verbose:
            if modified:
                print(f"[padding] modified {len(modified)} Conv2d layers -> '{pm}':")
                for name, old_mode, ks, pd in modified:
                    print(f"  - {name}: k={ks}, pad={pd}, '{old_mode}' -> '{pm}'")
            else:
                print("[padding] no Conv2d required modification.")

    # Optional public helper if you change config at runtime and want to reapply
    def reapply_padding_mode(self, log_changed: bool = False):
        self._apply_padding_mode(verbose=log_changed)
