## 6 Hours of Real-Life Streaming Data from the Forest

While initial model development and testing were done using the freesound dataset described previously, 6 hours of real-life audio data was obtained and used for model evaluation.

These audio clips, each 1 hour long, come from an audio streamer set up by the **Sensing the Forest** project. These were each split into 721 5-second segments, mirroring the length used in the final Freesound dataset that the model was trained on.

### Labeling
Following the length split, each of the clips were manually labeled, allowing for accuracy results from the models. Clips were labeled with either *speech*/*non_speech*/*flagged*. If speech was obvious, the clip was labeled speech, but in cases where speech was obscured or not completely obvious (such as yelling in the background, or indistinct conversation), the clip was flagged.

### Dataset Access
The final labeled five-second dataset and corresponding metadata CSV file can be found at: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16883289.svg)](https://doi.org/10.5281/zenodo.16883289)
