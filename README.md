# Few Shot Recognition

### Few Shot

Few shot classification refers to the problem of using a (pretrained) model to learn to classify a set of images with access to only a few examples of each class.

CLIP provides a powerful paradigm for this type of problem. It learns to associate text to images by embedding them in a common space. One can then construct a classifier by providing a set of possible captions that mention the possible labels for an image and select the caption with the highest similarity to the image.

There are several ways to generate captions for each image. One standard way is to use the phrase ‘a sentence of a {object}’ where {object} is replaced by the possible class names. Adding words to disambiguate similar objects sometimes brings about an improvement, as does using multiple prompts and using their mean as an embedding. For simplicity, in this work the simple prompt mentioned above was used.

We can finetune CLIP in several ways: using the entire the network end-to-end or merely the image encoder.

The code in this repo was based on the official CLIP repository as well as https://github.com/openai/CLIP/issues/83.

### Results

Experiments were done on a small subset of data from COCO. Without finetuning CLIP’s top-1 accuracy on the few-shot test data is 89.2% which is a formidable baseline.

The best finetuning performance was 91.3% after 24 epochs of training using a learning rate of 1e-7 and weight decay of 0.0001. Using higher learning rates and a higher weight decay in line with the values mentioned in the paper generally led to slightly worse results. Overall, it was observed that no hyperparameter combination yielded a performance significantly above the baseline, suggesting that CLIP pretraining is quite powerful for the classes in the few-shot dataset provided.

### Future Work

Beyond the scope of this task several methods to improve performance would be worth investigating:

- Adding image augmentations to improve robustness.
- In this work the training and testing data was used as is after a quick look over it suggested no obviously bad data. A natural next step would be to investigate this more closely and ensure there’s no mislabeled examples or ambiguity affecting the performance.
- When deploying the model we can accelerate inference by pre-computing the text embeddings and using those as a linear classification head since the labels we are interested in are known in the few-shot case.
- More complex training regimes using curriculum learning, learning rate schedulers, etc. would be worth trying for further improvement. The paper [Robust fine-tuning of zero-shot models](https://arxiv.org/abs/2109.01903) provides further suggestions on how to robustly combine zero-shot and fine-tuned models to improve accuracy on new data while remaining as accurate as possible on old data.

### Zero-Shot Learning with Unknown Classes

In some cases we might want to classify objects never encountered before. In particular, if we neither have training examples, nor know what classes to expect we effectively want to conduct anomaly detection and automatically update the database of known classes to include the new classes encountered. This is a challenging problem and an active field of research.

There are several ways we can approach this problem.

1. We could naively retry the approach described in the Few-Shot Classification section but provide a larger variety of prompts. Assuming our training data does not include precise examples of what we are interested in but sufficient amounts of adjacent classes of objects, we might be able to detect these new objects as well. Naturally, this approach is limited and does not scale well to large numbers of complex classes.
2. We could estimate the probability of an image belonging to a new class by estimating how uncertain a prediction is w/r/t the known classes. If e.g. all 8 classes from the few-shot dataset are assigned a probability of 12.5% our prediction will be no better than random guessing. We could use approaches such as [Monte Carlo Dropout](http://proceedings.mlr.press/v48/gal16.pdf) or deep ensembling to estimate uncertainty. If uncertainty is greater than a certain amount we could assign inputs with similar embeddings a new class type.
3. We can try to perform clustering in the latent space and whenever an image is too far away from the nearest cluster establish a new cluster center. Deciding when an input is too far away from existing clusters is a challenging problem.


## Usage

To finetune, follow the instructions to set up CLIP on the repo's homepage and then run `python few_shot_finetuning.py`.