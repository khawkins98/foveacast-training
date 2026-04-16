"""MSI-Net architecture in PyTorch.

A layer-by-layer port of Kroner et al.'s MSI-Net saliency model to PyTorch.

Upstream reference:
    Kroner, A., Senden, M., Driessens, K., & Goebel, R. (2020).
    "Contextual Encoder-Decoder Network for Visual Saliency Prediction."
    Neural Networks 129, pp. 261-270.
    https://doi.org/10.1016/j.neunet.2020.05.004
    Preprint: https://arxiv.org/abs/1902.06634
    Reference implementation (MIT): https://github.com/alexanderkroner/saliency

The port is intentionally verbatim: every layer has the same shape, the same
activation, the same padding, and the same dilation rate as the reference. The
preprocessing quirks (RGB input with BGR-ordered-by-convention mean subtraction)
are replicated exactly because Kroner's pretrained SALICON weights were trained
against this specific preprocessing, not the textbook VGG16 one. Numerical
parity against the upstream forward pass is the gate for Phase 2 per #1, so
every deliberate choice here exists to serve that parity check.

Architectural components:
    * Encoder: VGG16's 13 conv layers, with the last two maxpools reduced to
      stride=1 (so spatial resolution halts at H/8) and the last three convs
      dilated by rate=2 to compensate for the removed downsampling.
    * ASPP:    5 parallel branches (1x1 conv, and 3x3 convs at dilations 4/8/12,
      plus a global-average-pool-then-1x1-conv branch) concatenated and
      projected back to 256 channels via a 1x1 conv.
    * Decoder: 3 blocks of bilinear upsample + 3x3 conv (128 -> 64 -> 32 ch),
      followed by a final 3x3 conv to 1 channel with NO ReLU.
    * Output:  per-image min-max normalised to [0, 1].

Input contract (for callers and for the ONNX export in Phase 8):
    * shape: (N, 3, H, W) with H and W divisible by 8
    * dtype: float32
    * channel order: RGB
    * value range: [0, 255], NOT [0, 1]
    * mean subtraction: do not pre-apply — the graph does it internally

Output contract:
    * shape: (N, 1, H, W)
    * dtype: float32
    * value range: [0, 1] per-image (min-max normalised)

See ARCHITECTURE.md for how this module fits into the overall pipeline and
LEARNINGS.md (2026-04-16) for why we ported instead of vendoring Keras.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# why: hoisted as a module constant so the weight importer and the parity test
# can reference the exact same values. Order is (R, G, B) — matching the
# channel order of RGB inputs — but the values are VGG16's BGR-convention means
# applied to RGB channels. Non-standard but intentional: Kroner's pretrained
# weights were fit against this, so replicating it exactly is what gives us
# parity with the reference forward pass.
IMAGENET_MEAN_RGB_ORDER = (103.939, 116.779, 123.68)


class MSINet(nn.Module):
    """Kroner et al. 2020 MSI-Net, ported to PyTorch.

    Parameters
    ----------
    eps : float
        Small constant added to the divisor in the output normaliser to avoid
        division-by-zero on flat saliency maps. Matches the reference's 1e-7.
    """

    def __init__(self, eps: float = 1e-7) -> None:
        super().__init__()
        self.eps = eps

        # --- Encoder (VGG16 backbone) ------------------------------------
        # why: every Conv2d here uses kernel_size=3, stride=1, padding=1 which
        # reproduces TF's padding="same" for odd kernels. ReLU is applied
        # inline via F.relu in forward() rather than as a nn.ReLU module so
        # the parameter count of each Sequential matches Kroner's 1:1 (useful
        # when the importer walks the tf.keras.Model's weight list in order).

        # Block 1: 64 channels, ends with downsampling maxpool.
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # Block 2: 128 channels.
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        # Block 3: 256 channels. Output of this block (after pool3) is the
        # first of the three feature maps concatenated at encoder output.
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        # Block 4: 512 channels. Pool4 has stride=1 (NOT 2) — so spatial
        # resolution stays at H/8 after this block. Second of the three
        # feature maps in the encoder concatenation.
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        # Block 5: 512 channels, DILATED at rate=2 to compensate for the
        # removed downsampling at pool4/pool5. Pool5 also has stride=1.
        # Third of the three feature maps in the encoder concatenation.
        # why: padding = dilation for a 3x3 conv preserves spatial size.
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2)

        # --- ASPP --------------------------------------------------------
        # Five parallel branches that all produce (N, 256, H/8, W/8) feature
        # maps. The encoder output is (N, 256+512+512, H/8, W/8) = 1280 channels.
        self.aspp_b1 = nn.Conv2d(1280, 256, kernel_size=1)
        self.aspp_b2 = nn.Conv2d(1280, 256, kernel_size=3, padding=4, dilation=4)
        self.aspp_b3 = nn.Conv2d(1280, 256, kernel_size=3, padding=8, dilation=8)
        self.aspp_b4 = nn.Conv2d(1280, 256, kernel_size=3, padding=12, dilation=12)
        # Global-context branch: mean-pool over spatial dims → 1x1 conv → upsample.
        self.aspp_b5 = nn.Conv2d(1280, 256, kernel_size=1)
        # Projection after concatenating all five branches (256*5 = 1280 → 256).
        self.aspp_proj = nn.Conv2d(1280, 256, kernel_size=1)

        # --- Decoder -----------------------------------------------------
        # Three bilinear-upsample-then-3x3-conv blocks, channel progression
        # 256 → 128 → 64 → 32, followed by a final 3x3 conv to 1 channel
        # with NO ReLU (the output is later min-max normalised).
        self.decoder_conv1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.decoder_conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.decoder_conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.decoder_conv4 = nn.Conv2d(32, 1, kernel_size=3, padding=1)

    # ---------- forward passes, one per architectural component ----------

    def _encoder(self, x: torch.Tensor) -> torch.Tensor:
        """VGG16 encoder with the last two maxpools reduced to stride=1 and
        block 5 dilated at rate=2. Returns a (N, 1280, H/8, W/8) feature map
        formed by concatenating pool3, pool4, and pool5 activations.
        """
        # Block 1 → pool1 (stride 2, so H becomes H/2)
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        # Block 2 → pool2 (stride 2, so H becomes H/4)
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        # Block 3 → pool3 (stride 2, so H becomes H/8). pool3 output kept
        # for the encoder concat.
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        pool3 = F.max_pool2d(x, kernel_size=2, stride=2)

        # Block 4 → pool4. pool4 has stride=1 and TF "same" padding; for
        # a 2x2 kernel at stride=1 this means pad right+bottom by 1 to keep
        # spatial dims. PyTorch's MaxPool2d has no "same" mode, so we pad
        # explicitly with F.pad before the pool. why: exact parity with
        # Kroner's ops matters — symmetric padding or no padding would
        # shift the output by up to a pixel, compounding through the
        # decoder upsamples.
        x = F.relu(self.conv4_1(pool3))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        pool4 = F.max_pool2d(F.pad(x, (0, 1, 0, 1)), kernel_size=2, stride=1)

        # Block 5 (dilated) → pool5 (also stride=1 with right+bottom pad).
        x = F.relu(self.conv5_1(pool4))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))
        pool5 = F.max_pool2d(F.pad(x, (0, 1, 0, 1)), kernel_size=2, stride=1)

        # Concatenate three scales: pool3 (256), pool4 (512), pool5 (512) = 1280.
        return torch.cat([pool3, pool4, pool5], dim=1)

    def _aspp(self, features: torch.Tensor) -> torch.Tensor:
        """Atrous Spatial Pyramid Pooling. Five parallel branches plus a
        concat-and-project. Preserves input spatial dimensions.
        """
        b1 = F.relu(self.aspp_b1(features))
        b2 = F.relu(self.aspp_b2(features))
        b3 = F.relu(self.aspp_b3(features))
        b4 = F.relu(self.aspp_b4(features))

        # Global-context branch: collapse spatial dims via mean, 1x1 conv,
        # then upsample back to the input spatial shape. keepdim=True on
        # the mean so the subsequent conv sees a (N, C, 1, 1) tensor, as
        # the reference does.
        b5 = features.mean(dim=(2, 3), keepdim=True)
        b5 = F.relu(self.aspp_b5(b5))
        b5 = F.interpolate(
            b5,
            size=features.shape[2:],
            mode="bilinear",
            # why: align_corners=False matches TF 1.x tf.image.resize_bilinear
            # default, which is what Kroner's reference used at training time.
            align_corners=False,
        )

        context = torch.cat([b1, b2, b3, b4, b5], dim=1)
        return F.relu(self.aspp_proj(context))

    def _decoder(self, features: torch.Tensor) -> torch.Tensor:
        """Three upsample-and-conv blocks followed by a single-channel head.
        Output spatial dimensions are 8x the input (recovering the H/8
        downsampling the encoder did).
        """
        # why: each block doubles spatial resolution via bilinear upsample,
        # then applies a 3x3 conv. align_corners=False matches TF 1.x default,
        # same rationale as in _aspp.
        x = F.interpolate(features, scale_factor=2, mode="bilinear", align_corners=False)
        x = F.relu(self.decoder_conv1(x))

        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = F.relu(self.decoder_conv2(x))

        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = F.relu(self.decoder_conv3(x))

        # Final conv to single channel — NO ReLU. The output passes through
        # _normalize next, which min-max scales to [0, 1] per image.
        return self.decoder_conv4(x)

    def _normalize(self, maps: torch.Tensor) -> torch.Tensor:
        """Per-image min-max normalisation to [0, 1]. Matches the reference's
        `_normalize` step: subtract the per-image minimum, divide by the
        per-image maximum plus eps.
        """
        # why: reduce across channel + spatial dims (everything except batch)
        # so every image in the batch is scaled independently.
        reduce_dims = (1, 2, 3)

        min_per_image = maps.amin(dim=reduce_dims, keepdim=True)
        maps = maps - min_per_image

        max_per_image = maps.amax(dim=reduce_dims, keepdim=True)
        return maps / (self.eps + max_per_image)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Forward pass: RGB [0, 255] image batch → normalised saliency map.

        Parameters
        ----------
        images : torch.Tensor
            Shape (N, 3, H, W), dtype float32, RGB channel order, value
            range [0, 255]. H and W must be divisible by 8 (the encoder
            downsamples 3 times before ASPP).

        Returns
        -------
        torch.Tensor
            Shape (N, 1, H, W), dtype float32, per-image min-max normalised
            to [0, 1].
        """
        # why: mean subtraction is done inside the graph so that callers
        # (including the ONNX consumer in Foveacast) don't need to know the
        # specific Kroner-quirky mean values. This also makes the ONNX
        # artefact self-contained — no JS-side footgun risk.
        mean = torch.tensor(
            IMAGENET_MEAN_RGB_ORDER,
            dtype=images.dtype,
            device=images.device,
        ).view(1, 3, 1, 1)
        images = images - mean

        features = self._encoder(images)
        context = self._aspp(features)
        logits = self._decoder(context)
        return self._normalize(logits)


def load_pretrained_msinet(weights_path: str, eps: float = 1e-7) -> MSINet:
    """Instantiate MSINet and load weights from a .pt state_dict file.

    Parameters
    ----------
    weights_path : str
        Path to a state_dict produced by `scripts/import_msinet_weights.py`.
    eps : float
        Passed through to MSINet's constructor.

    Returns
    -------
    MSINet
        In eval mode, with weights loaded.
    """
    model = MSINet(eps=eps)
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    # why: strict=True catches any layer-name mismatch between the importer
    # and this module early — better to fail loud than silently load a
    # half-populated network.
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


if __name__ == "__main__":
    # Smoke test: instantiate the model, run a random (240, 320) input
    # through it, and print shapes. This does not exercise weight loading
    # or parity; that is the parity test's job.
    model = MSINet()
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"MSINet parameters: {n_params:,}")

    x = torch.randn(1, 3, 240, 320) * 127 + 128  # fake "image" in [0, 255]
    with torch.no_grad():
        y = model(x)
    print(f"input shape:  {tuple(x.shape)}")
    print(f"output shape: {tuple(y.shape)}")
    print(f"output range: [{y.min().item():.4f}, {y.max().item():.4f}]")
