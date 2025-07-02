# MFCC Support Vector Machine (SVM) Progression

A 2012 paper identified **Mel-Frequency Cepstral Coefficients (MFCCs)** combined with a **Support Vector Machine (SVM)** can be effective for voice activity detection. Following the paper, in this first attempt, **12 MFCCs** were extracted for each audio clip. An SVM was then trained on a combined **train + validation** dataset (following the same split as outlined in the `dataset` section). While this is a relatively simple model, it lays the groundwork for more advanced approaches. Despite this, this first model resulted in strong performance on the held-out test dataset (the original full-length dataset):
- **Accuracy**: 0.796
- **Precision**: 0.745
- **Recall**: 0.884
- **F1 Score**: 0.809

This result is promising, especially for a lightweight model. The precision–recall balance indicates that the classifier correctly detects most speech clips while keeping false positives relatively low. While it doesn't reach perfect recall, the model captures the majority of speech instances and can serve as a solid foundation for further tuning or integration into an ensemble system.

# Iterative Improvements
To improve from the baseline, the same paper includes two easy steps: scaling and delta + delta-delta features.

### 1. Feature Scaling  
SVMs are sensitive to feature scale, so normalization was introduced using a standard scaler. This alone yielded a marked improvement:

- **Accuracy**: 0.869  
- **Precision**: 0.864  
- **Recall**: 0.884  
- **F1 Score**: 0.874  

### 2. Delta and Delta-Delta MFCC Features  
To enhance the representation of speech dynamics, **Delta** (first derivative) and **Delta-Delta** (second derivative) features were added. These capture how the MFCCs change over time, increasing the total features per clip from 12 to 36. Combined with normalization, this led to further gains:

- **Accuracy**: 0.923  
- **Precision**: 0.911  
- **Recall**: 0.954  
- **F1 Score**: 0.932  

While precision slightly decreased, the significant boost in recall is valuable for this task. In the context of VAD, **false negatives (missed speech)** are more harmful than **false positives**, so a higher recall is preferable.

### 3. Feature Enhancement
To improve the robustness of the features and add a new dimension, the **standard deviation** of the MFCC features were included as well as the mean that was used initially, bringing the total features per clip to 72. Stacking with the previous two iterations, this came with marked improvements:

- **Accuracy**: 0.976 
- **Precision**: 0.956 
- **Recall**: 1.000 
- **F1 Score**: 0.977

This model **achieved perfect recall** while maintaining high precision, making it highly suitable for robust speech detection.

### Interation Comparison Table

| Model Version             | Features Used                | Scaling | Accuracy  | Precision | Recall    | F1 Score  |
|---------------------------|------------------------------|---------|-----------|-----------|-----------|-----------|
| **Base MFCC SVM**         | 12 MFCC                      | ❌       | 0.796     | 0.745     | 0.884     | 0.809     |
| **+ Scaler**              | 12 MFCC                      | ✅       | 0.869     | 0.864     | 0.884     | 0.874     |
| **+ Delta + Delta-Delta** | 12 MFCC + Δ + ΔΔ (total: 36) | ✅       | 0.923     | 0.911     | 0.954     | 0.932     |
| **+ Std Feature**         | 12 MFCC + Δ + ΔΔ (total: 72) | ✅       | **0.976** | **0.956** | **1.000** | **0.977** |

## New Dataset, and "Realistic" Dataset Testing
Following the creation of the 5-second dataset (see dataset section for more information), the final iteration of the SVM model was tested again. This time, testing was done between 12 and 13 MFCC features. Both results are extremely strong, and very similar, but the 13 MFCC version was chosen due to the marginally higher **accuracy, precision,** and **F1 score**.

| # of MFCCs | Accuracy  | Precision | Recall | F1 Score  |
|------------|-----------|-----------|--------|-----------|
| **12**     | 0.973     | 0.962     | 0.980  | 0.971     |
| **13**     | **0.977** | **0.971** | 0.980  | **0.976** |

### Realistic Dataset
The final model was tested using a "realistic" dataset, that contained a training split that was 10%/90% speech/non_speech. In the case of this dataset, the training subset was made up of 130 audio clips, 13 of which were labeled speech. This was done to mimic the final conditions that the model will be used in, an environment without much speech. Across 50 randomized runs, the model consistently achieves 97% accuracy, 97.7% recall, and 88% F1 score.

The stats from the 50 runs are shown below:

|        | Accuracy | Precision | Recall   | F1 Score |
|--------|----------|-----------|----------|----------|
| **Mean**   | **0.974615** | **0.808750** | **0.976923** | **0.884729** |
| Std    | 0.003561 | 0.005786  | 0.035608 | 0.018243 |
| Min    | 0.969231 | 0.800000  | 0.923077 | 0.857143 |
| 25%    | 0.969231 | 0.800000  | 0.923077 | 0.857143 |
| 50%    | 0.976923 | 0.812500  | 1.000000 | 0.896552 |
| 75%    | 0.976923 | 0.812500  | 1.000000 | 0.896552 |
| Max    | 0.976923 | 0.812500  | 1.000000 | 0.896552 |

The results show that the worst the model did in 50 attempts was miss **1 out of 13** speech clips, and averaged a success rate of **12.7 out of 13** speech clips correctly identified. 

The precision is potentially a little low, **~80%** means that **20% of the clips the model identified as speech was actually non-speech**. While this can certainly be addressed, within the context of a VAD focusing on high recall (successfully catching as much speech as possible) is generally more important that a high precision (over-estimating some non-speech as speech). Future iterations should look for improvements in precision, but not at the expense of much recall.