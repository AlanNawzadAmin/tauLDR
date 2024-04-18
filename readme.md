# A Continuous Time Framework for Discrete Denoising Models
[**Paper Link**](https://arxiv.org/abs/2205.14987v2)

## Install

```bash
conda env create --file tldr_env.yml
```


## Sampling

To reproduce the FID score reported in the paper, first download the pre-trained model from https://www.dropbox.com/scl/fo/zmwsav82kgqtc0tzgpj3l/h?dl=0&rlkey=k6d2bp73k4ifavcg9ldjhgu0s

Then go to `config/eval/cifar10.py` and change the `model_location` field to point to the cifar10 pytorch checkpoint.
Also change `model_config_location` to the config included with the pytorch checkpoint.

We provide a sampling script to sample 50000 images from the model. 
Go to `scripts/sample.py`, and change `save_samples_path` to a location where samples can be saved.
Then run
```bash
python scripts/sample.py
```

Then to compute the FID score, you will also need to download the CIFAR10 dataset.
Once you have the CIFAR10 pngs in another folder you can compute the FID score using
```bash
python -m pytorch_fid --device cuda:0 path/to/tauLDR_samples path/to/cifar10pngs
```

### ELBO evaluation
To reproduce the ELBO value reported in the paper for the CIFAR10 model, obtain the checkpoint as above.

Then go to `config/eval/cifar10_elbo.py` and change `config.experiment_dir` to point to a directory that has the following structure
```
experiment_dir
    - checkpoints
        - ckpt_0001999999.pt
    - config
        - config_001.yaml
```
using the files downloaded from the dropbox. Also change `config.checkpoint_path` to point to `/path/to/experiment_dir/checkpoints/ckpt_0001999999.pt`.
Change `config.cifar10_path` to point to somewhere where the CIFAR10 dataset can be downloaded (it will be automatically downloaded to that location).

Finally, run the following command
```
python elbo_evaluation.py
```
This will save the ELBO value in a eval folder within the experiment_dir. The ELBO value is written in the file `neg_elbo`, the first 0 can be ignored and the second number is the negative ELBO averaged over the pixels. It should be around 3.59.

## Notebooks

To generate CIFAR10 samples, open the `notebooks/image.ipynb` notebook.
Change the paths at the top of  the `config/eval/cifar10.py` config file to point to a folder where CIFAR10 can be downloaded and the paths to the model and config downloaded from the dropbox link. 

To generate piano samples, open the `notebooks/piano.ipynb` notebook.
Change the paths at the top of the `config/eval/piano.py` config file to point to the dataset downloaded from the dropbox link as well as the model weights and config file.

The sampling settings can be set in the config files, switching between standard tau-leaping and with predictor-corrector steps.


## Training
### CIFAR10
The CIFAR10 model can be trained using
```
python train.py cifar10
```
Paths to store the output and to download the CIFAR10 dataset should be set in the training config, `config/train/cifar10.py`.
To train the model over multiple GPUs, use
```
python dist_train.py cifar10
```
with settings found in the `config/train/cifar10_distributed.py` config file.

### Piano
The piano model can be trained using
```
python train.py piano
```
Paths to store the output and to the dataset downloaded from the dropbox link should be set in `config/train/piano.py`.


## Audio Samples
These are 4 pairs of audio samples. The first is a music sequence generated by the model conditioned on the first 2 bars (~2.5 secs) of the piece. The second is the ground truth song from the test dataset.

### Pair a
https://user-images.githubusercontent.com/72878071/170738786-d817c70e-aaab-4ed1-ac8d-ffde3e95389a.mp4

https://user-images.githubusercontent.com/72878071/170738800-03794049-c114-45ae-b1ce-d205f390b622.mp4

### Pair b
https://user-images.githubusercontent.com/72878071/170738886-414cf245-0e22-4e3f-91ef-45bb8029f842.mp4

https://user-images.githubusercontent.com/72878071/170738904-29021d7b-6416-4462-9e51-ac4dc5dab3a6.mp4

### Pair c
https://user-images.githubusercontent.com/72878071/170738931-69d97623-5368-4ff5-8f2a-a98da29fa4e3.mp4

https://user-images.githubusercontent.com/72878071/170738963-353e4bee-c290-4418-b62f-d9d34b731a65.mp4

### Pair d
https://user-images.githubusercontent.com/72878071/170738998-7f6ad1e3-ea3b-49af-9d29-938aabed3544.mp4

https://user-images.githubusercontent.com/72878071/170739023-c9b15acb-45f1-48cf-a02c-e25f4e76d2f6.mp4

