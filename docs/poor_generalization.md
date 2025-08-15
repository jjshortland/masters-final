## Model Testing

The successful model was tested against the real-world dataset. Despite having good results within the freesound dataset it was trained on, the model did not generalize well, as shown in the table below:

| Dataset        | Accuracy  | Precision | Recall | F1 Score  |
|----------------|-----------|-----------|--------|-----------|
| **Freesound**  | **0.977** | **0.971** | 0.980  | **0.976** |
| **Real-world** | 0.957     | 0.919     | 0.366  | 0.523     |

While the accuracy is still very high, context is required. The real-world dataset is majority non-speech (~88%). Therefore, the model's high accuracy could largely be down to its ability to identify non-speech clips. The precision being high is a good sign, suggesting that when the model predicts speech, it is generally correct. However, the model falls apart at recall. 37% recall suggests that while it typically gets speech correct when labeling it, the model is missing the majority of the speech files. As a result, the F1 score is much lower as well.