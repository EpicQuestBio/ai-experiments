Jade's experiments running neural nets.

The script run_QuipuNet.py is a modern version of the QuipuNet repository's (https://github.com/kmisiunas/QuipuNet/) Colab demo that runs locally on the dataset they used. It doesn't need much in the way of resources and I was even able to run it on CPU because I have an AMD GPU and didn't want to install TensorFlow-DirectML just for this.

jreremy_conformer_huggingface_script.py is a HuggingFace "jobs uv run" - compatible script that replicates a simplified version of the Conformer paper's LibriSpeech experiment. It uses the https://github.com/jreremy/conformer which has a unique implementation of Conformer with some differences and much custom code for applying it to LibriSpeech.

My next step was to use https://github.com/sooftware/conformer on the nanopore data in the QuipuNet repository! It's the Bell & Keyser dataset which contains current traces from a solid-state nanopore sensing DNA barcodes sometimes attaching to a specific protein and identifying the protein!

On a grouped-dev, held-out-experiment split, a Conformer classifier outperformed the original QuipuNet CNN overall (85.4% vs 81.0% test accuracy), while both models showed a pronounced failure mode on barcode 101, suggesting that this class is especially sensitive to cross-experiment shift rather than the issue being specific to one architecture.

The script hf_quipunet_conformer_uv.py runs Conformer on the B&K dataset. run_QuipuNet_same_conditions.py can be copied into the QuipuNet repository clone to run QuipuNet with the same conditions. conformer_quipunet_metrics.py is a metrics script that loads a Conformer and provides detailed diagnostics about barcode 101.
