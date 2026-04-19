Jade's experiments running neural nets.

The script run_QuipuNet.py is a modern version of the QuipuNet repository's (https://github.com/kmisiunas/QuipuNet/) Colab demo that runs locally on the dataset they used. It doesn't need much in the way of resources and I was even able to run it on CPU because I have an AMD GPU and didn't want to install TensorFlow-DirectML just for this.

jreremy_conformer_huggingface_script.py is a HuggingFace "jobs uv run" - compatible script that replicates a simplified version of the Conformer paper's LibriSpeech experiment. It uses the https://github.com/jreremy/conformer which has a unique implementation of Conformer with some differences and much custom code for applying it to LibriSpeech.

My next step is to use https://github.com/sooftware/conformer on the nanopore data in the QuipuNet repository! It's the Bell & Keyser dataset which contains current traces from a solid-state nanopore sensing DNA barcodes sometimes attaching to a specific protein and identifying the protein!
