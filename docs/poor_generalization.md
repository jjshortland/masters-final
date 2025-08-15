## Model Testing

The successful model was tested against the real-world dataset. Despite having good results within the freesound dataset it was trained on, the model did not generalize well, as shown in the table below:

| Dataset        | Accuracy  | Precision | Recall    | F1 Score  |
|----------------|-----------|-----------|-----------|-----------|
| **Freesound**  | **0.977** | **0.971** | **0.980** | **0.976** |
| **Real-world** | 0.957     | 0.919     | 0.366     | 0.523     |

While the accuracy is still very high, context is required. The real-world dataset is majority non-speech (~88%). Therefore, the model's high accuracy could largely be down to its ability to identify non-speech clips. The precision being high is a good sign, suggesting that when the model predicts speech, it is generally correct. However, the model falls apart at recall. 37% recall suggests that while it typically gets speech correct when labeling it, the model is missing the majority of the speech clips. The F1 score is much lower as a result.

### Training on the Real-World Dataset

In order to attempt to improve these results, models were trained on the real-world data and tested. These were done across three iterations:

- **Balanced** training and testing datasets
- **Balanced** training and **unbalanced** testing datasets
- **Unbalanced** training and testing datasets

| Training Dataset | Testing Dataset | Accuracy  | Precision | Recall    | F1 Score  |
|------------------|-----------------|-----------|-----------|-----------|-----------|
| **Balanced**     | **Balanced**    | 0.794     | **0.982** | **0.598** | **0.743** |
|  **Balanced**    | **Unbalanced**  | 0.953     | 0.458     | **0.598** | 0.519     |
| **Unbalanced**   | **Unbalanced**  | **0.962** | 0.556     | 0.489     | 0.520     |

Initially, a balanced dataset was built for both training and testing. However, considering the unbalanced nature of the final dataset, balancing the testing dataset made little sense. Therefore, future tests were all done with an unbalanced test set. The resulting models that were trained on balanced and unbalanced datasets respectively had interesting results. Although they had similar accuracy and F1 scores, the breakdown of the F1—precision and recall—is flipped.

### Pre-Trained VADs

In another attempt to work with this data, the pre-trained VAD were retested. The two best performing models, Silero and WebRTC from the previous dataset were used in this instance. Because they didn't need to be trained, both models were tested on the entire real-world dataset. The result are shown in the table below:

| VAD        | Accuracy  | Precision | Recall    | F1 Score  |
|------------|-----------|-----------|-----------|-----------|
| **Silero** | **0.956** | **0.887** | 0.394     | **0.546** |
| **WebRTC** | 0.529     | 0.116     | **0.900** | 0.201     |

The results are interesting, and provided an idea of what the next steps could be. WebRTC had the highest recall of the models tested, meaning it successfully labeled 90% of the speech clips. However, some context is needed. The extremely low precision score of 11.6% means that the majority of the clips that it labels as speech are non-speech. Silero, on the other hand, had similar results to the bespoke trained models: a high precision score indicating a high confidence in the clips it identified as speech. Because the models seem to complement each other well, they worked well as candidates for a two-model filter system.

The premise is simple: of the clips that WebRTC identifies as speech, the vast majority might be non-speech, but most of the speech clips will still be present. That means that any clips that it does not identify as speech are almost certainly not speech (minus 10% margin for error). Therefore, it works well as a first pass filter before Silero can look at the data. In theory, this balances the high recall from the WebRTC, 'giving' Silero a smaller collection of clips that contain most of the speech. Silero, with its higher precision, could then narrow down the pool further and have more successful results overall.

### Two Filter VAD System
