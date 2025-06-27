# Masters Final Project

The final project for my MSc Data Science and Artificial Intelligence degree. In collaboration with the **Sensing the Forest** organization, this project will aim to take a livestreamed audio soundscape of a forest and remove human voices, before publishing it to the website for the public to enjoy. There are a number of steps that go into a project like this, and details of each step can be found below.

## 1. [Dataset](dataset.md)
First, a dataset needs to be constructed that models the data that the model will receive. In this case, the dataset needed to contain a selection of audio clips, either containing sounds of the forest, or containing speech. Ultimately, the solution to this problem is one of binary classification: can the model identify if there is speech in a given audio clip?
[Dataset Overview](dataset.md)

## 2. [Pre-Trained Voice Activity Detectors](pre_trained_vad_testing.md)
There are a number of pre-trained ready-to-use voice activity detectors available. I picked three, and compared them against the dataset.
[Testing of Pre-trained Voice Activity Detectors](pre_trained_vad_testing.md)

## 3. [Bespoke Model: Support Vector Machine](SVM.md)
After experimenting with pre-trained models, I decided to train my own. Drawing on the approach outlined in a 2012 paper, which identified Mel-Frequency Cepstral Coefficients (MFCCs) combined with a Support Vector Machine (SVM) as an effective method for voice activity detection, I implemented and evaluated my own version of this technique.
[Support Vector Machine Model Progression](SVM.md)

