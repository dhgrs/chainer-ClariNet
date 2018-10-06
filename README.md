# chainer-ClariNet

A Chainer implementation of ClariNet( https://arxiv.org/abs/1807.07281 ).

# Results
## Autoregressive WaveNet(Single Gaussian ver.) trained with VCTK Corpus
https://nana-music.com/sounds/04027269/

## Student Gaussian IAF trained with LJ-Speech
https://nana-music.com/sounds/043ba7b4/

# Requirements
I trained and generated with

- python(3.5.2)
- chainer (5.0.0b4)
- librosa (0.6.2)
- matplotlib (2.2.3)
- tqdm (4.25.0)

# Usage
## download dataset
You can download VCTK Corpus(en multi speaker)/LJ-Speech(en single speaker) very easily via [my repository](https://github.com/dhgrs/download_dataset).

## set parameters
Almost parameters in `params.py` and `teacher_params.py` are same as `params.py` in my other repositories like [VQ-VAE](https://github.com/dhgrs/chainer-VQ-VAE). If you modified `params.py` in AutoregressiveWavenet, you have to replace `teacher_params.py` with it to train student.

## training
You can use same command in each directory.
```
(without GPU)
python train.py

(with GPU #n)
python train.py -g n
```

You can resume snapshot and restart training like below.(Now support AutoregressiveWaveNet only)
```
python train.py -r snapshot_iter_100000
```
Other arguments `-f` and `-p` are parameters for multiprocess in preprocessing. `-f` means the number of prefetch and `-p` means the number of processes. I highly recommend to modify `-f` to large number like `64`. If GPU-Util is stil low, modify `-p` to large number like `8`.

## generating
```
python generate.py -i <input file> -o <output file> -m <trained model>
```

If you don't set `-o`, default file name `result.wav` is used. If you don't set `-s`, the speaker is same as input file that got from filepath.

# Caution
I only check the results for 

- Autoregressive WaveNet(Single Gaussian ver.)
- Student Gaussian IAF

