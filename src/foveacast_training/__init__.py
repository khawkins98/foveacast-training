"""foveacast-training — saliency-model training pipeline for Foveacast.

The package is deliberately thin at this stage — scaffolding only. Training
code (``msinet``, ``ueyes_dataset``, ``train``, ``eval``, ``export_onnx``)
lands in subsequent commits once the dataset loader has been verified
against the actual UEyes bytes.

See ``README.md`` for the project's purpose, provenance chain, and
reproduction notes; see ``LEARNINGS.md`` for the dated decision log.
"""

__version__ = "0.1.0"
