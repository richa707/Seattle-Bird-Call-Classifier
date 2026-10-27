# Seattle Bird Call Classifier

Custom convolutional neural networks (CNNs) that identify 12 bird species common to Seattle from audio spectrograms. Includes binary and 12-class models, a comparison of several architectures, and predictions on three unlabeled mp3 clips.

> Course project for DATA 5322 (Deep Learning), Seattle University, 2025.

## Overview

Bird calls are converted into spectrograms, 128 (frequency) × 517 (time) "images" of sound. A CNN treats them like images and learns the frequency patterns that separate species.

1. **Binary model:** American Robin vs. Red-winged Blackbird.
2. **Multi-class model:** all 12 species.
3. **External test:** 3 unlabeled raw mp3 clips, checked for one vs. multiple birds.

**Species:** American Crow, American Robin, Bewick's Wren, Black-capped Chickadee, Dark-eyed Junco, House Finch, House Sparrow, Northern Flicker, Red-winged Blackbird, Song Sparrow, Spotted Towhee, White-crowned Sparrow.

## Key Findings

- **Binary:** the best CNNs reach about **74% held-out accuracy** (AUC about 0.82 to 0.86), against a roughly 52% majority-class baseline. Training accuracy was often near 100%, so overfitting was the main issue.
- **12-class:** the best full-dataset model reaches **34.5% held-out accuracy**, which is weak. Always predicting House Sparrow would score about 31.9% (158 of 496), so this model barely beats that baseline. It predicts House Sparrow for most inputs.
- **External clips:** two of three clips were predicted to contain two species, but there is no ground truth, and the model's bias toward House Sparrow makes these predictions low-confidence.
- **Takeaway:** with roughly 1,900 samples, heavy class imbalance, and limited compute, the multi-class problem was much harder than the binary one.

## Data

- **Source:** Kaggle "Xeno-Canto Bird Recordings Extended (A-M)", originally from [Xeno-Canto](https://xeno-canto.org). The 12 species and the spectrograms were provided by the instructor.
- **Provided preprocessing:** audio subsampled to 22,050 Hz, first 3 seconds kept, spectrogram per 2-second window. Stored as HDF5, read with `h5py`.
- **My preprocessing:** per-sample min-max normalization, then a stratified 75/25 train/held-out split.
- **Class imbalance:** 37 to 630 samples per species (House Sparrow is the largest, Northern Flicker and Black-capped Chickadee the smallest). For the binary task I chose two species with similar counts (about 170 and 190 samples). The multi-class models did not use class weights or oversampling, which is listed under future work.
- The data file is not included because of its size. [Add a link or instructions for obtaining it.]

## Results

*All results use a single 75/25 split, so the held-out set serves as both validation and test data. Treat these numbers as slightly optimistic.*

### Binary: American Robin vs. Red-winged Blackbird

| Model | Architecture | Train Acc | Val Acc | Val AUC | Val Precision | Val Recall |
|---|---|---|---|---|---|---|
| I | 2 conv layers, dropout 0.3 | 1.000 | 0.700 | 0.806 | 0.679 | 0.809 |
| I + early stopping | same as I | 0.998 | 0.656 | 0.757 | 0.735 | 0.532 |
| II | 3 conv layers, dropout, fewer dense units | 0.993 | **0.744** | 0.820 | 0.688 | 0.936 |
| III | 3 conv layers, larger kernels | 0.857 | 0.733 | **0.856** | 0.735 | 0.766 |
| IV | 2 conv layers, average pooling | 0.965 | **0.744** | 0.820 | 0.773 | 0.723 |

Model III overfit the least (smallest train/validation gap) and had the best AUC. Models II and IV tied on accuracy. Early stopping on Model I did not help. Validation curves for every binary model fluctuate heavily from epoch to epoch while training curves climb smoothly, a typical sign of a small validation set and overfitting.

Confusion matrix for Model I: 32 of 43 robins and 33 of 47 blackbirds classified correctly.

### Multi-class: 12 species

| Model | Data | Train Acc | Val Acc | Train Loss | Val Loss |
|---|---|---|---|---|---|
| I | Subset | 0.683 | 0.450 | 1.020 | 1.829 |
| II | Full, early stopping | 0.530 | 0.434 | 1.467 | 1.748 |
| III | Full, early stopping | 0.339 | 0.099 | 2.064 | 3.327 |
| IV | Full, early stopping | 0.313 | 0.270 | 2.127 | 2.233 |
| V (final) | Full | 0.339 | **0.345** | 2.143 | 2.074 |

Model I used a data subset, so its accuracy is not comparable to the full-dataset models. Models III and IV [describe changes, e.g. LeakyReLU, average pooling, broader kernels]. Model V had the smallest train/validation gap and the best full-dataset accuracy, but it is still weak in absolute terms.

Most species are predicted as House Sparrow (the largest class), with a few predicted as Song Sparrow or American Robin. House Finch, Bewick's Wren, and White-crowned Sparrow were rarely predicted. House Sparrow is correct 156 of 158 times, but Song Sparrow only 13 of 66, so the strong House Sparrow score reflects the model's bias toward that class.

### External test clips

Each clip was split into segments, and each segment was classified separately. A clip is flagged as multi-bird when more than one species is predicted across its segments.

| Clip | Segments | Predicted Species (segment count) | Multiple Birds? |
|---|---|---|---|
| test1.mp3 | 9 | House Sparrow (8), Dark-eyed Junco (1) | Likely yes |
| test2.mp3 | 7 | House Sparrow (7) | Likely no |
| test3.mp3 | 9 | House Sparrow (6), Dark-eyed Junco (3) | Yes |

There is no ground truth for these clips, and the model favors House Sparrow, so treat the predictions as tentative. Song Sparrow and Dark-eyed Junco appearing together may point to similar spectrograms, but this was not verified.

## Approach

Every experiment, including the ones that performed poorly, is in the results tables above.

- **Binary:** four CNN variants varying depth, kernel size, dense units, and pooling type (max vs. average), plus one early-stopping run.
- **Multi-class:** five CNN variants, starting with a 3-layer CNN on a data subset and moving to the full dataset with larger kernels, early stopping, LeakyReLU, and average pooling.
- **Regularization:** dropout and early stopping (patience on validation metrics).
- **Training time:** binary models trained quickly. Each multi-class model took about 25 to 35 minutes on [your hardware].

## Why a CNN?

A spectrogram is a 2D frequency-by-time image, and CNNs are built to find local patterns in images, such as the shape and pitch contour of a call. RNNs could treat the audio as a sequence, but they train slowly and cannot be parallelized, so CNNs were the practical choice here. Classical models like SVMs or random forests would need hand-built audio features first.

## Limitations and Future Work

- **Hardware:** limited compute capped the number of epochs and the model sizes I could try.
- **Class imbalance:** 37 to 630 samples per species. The multi-class model collapses toward the largest class. Next steps: class weights, oversampling, and spectrogram augmentation (time shift, noise).
- **Evaluation:** a single train/held-out split with no separate test set. Cross-validation would give more reliable numbers.
- **External clips:** no ground truth, so multi-bird detection was not validated.
- **Other approaches:** transfer learning from pretrained audio/image models, longer or manually trimmed audio windows, and sliding-window prediction. RNNs and transformers were considered but not tested.
- **Species confusion:** House Finch, Bewick's Wren, and White-crowned Sparrow were rarely predicted. This may reflect similar calls or simply low sample counts, and I did not verify which.

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── notebooks/
│   └── [your notebook name].ipynb
├── report/
│   └── HW3_revised.pdf
├── figures/
└── .gitignore
```

## How to Run

```bash
git clone https://github.com/richa707/HW3.git
cd HW3
pip install -r requirements.txt
```

1. Download the spectrogram data and place it in `data/` (see Data above).
2. Open the notebook and run all cells.

**Tech stack:** Python, TensorFlow/Keras, NumPy, h5py, scikit-learn, Matplotlib

## References

- Xeno-Canto: https://xeno-canto.org (data via the Kaggle "Xeno-Canto Bird Recordings Extended (A-M)" dataset)
- IBM. (2021). What is a neural network? https://www.ibm.com/think/topics/neural-networks
- Sankaran, S. Understanding Convolutional Neural Networks (CNN). LearnOpenCV. https://learnopencv.com/understanding-convolutional-neural-networks-cnn/
- Das, K. (2020). How Recurrent Neural Network (RNN) Works. Dataaspirant. https://dataaspirant.com/how-recurrent-neural-network-rnn-works/
- Mendible, A. (2025). Deep Learning and Deep Learning II [Class notes]. DATA 5322, Seattle University.

## Author

**Richa Deshpande**, M.S. Data Science, Seattle University
