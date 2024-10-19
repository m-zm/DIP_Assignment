# Assignment 2 - DIP with PyTorch

## DIP with PyTorch

This repository is the implementation of Assignment_02 of DIP.

<img src="pics/teaser.png" alt="teaser" width="800">

## Requirements

To install requirements:

```setup
python -m pip install -r requirements.txt
```

## Running

To run poisson blending, run:

```poisson
python run_blending_gradio.py
```

To run Pix2Pix, run:

```Pix2Pix
python infer.py
```

## Results

### Poisson Blending

<img src="pics/poisson_1.gif" alt="poisson demo 1" width="800">
<img src="pics/poisson_2.gif" alt="poisson demo 2" width="800">

### Pix2Pix

<img src="pics/facade_1.png" alt="facade 1" width="800">
<img src="pics/facade_2.png" alt="facade 2" width="800">
<img src="pics/cityscape_1.png" alt="cityscape 1" width="800">
<img src="pics/cityscape_2.png" alt="cityscape 2" width="800">

## Acknowledgement

>📋 Thanks for the algorithms proposed by
[Poisson Image Editing](https://www.cs.jhu.edu/~misha/Fall07/Papers/Perez03.pdf),
[Pix2Pix](https://phillipi.github.io/pix2pix/) and
[Fully Convolutional Layers](https://arxiv.org/abs/1411.4038).
