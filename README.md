# Few Shot Recognition

### Few Shot

Few shot classification refers to the problem of using a (pretrained) model to learn to classify a set of images with access to only a few examples of each class. In the task at hand the training dataset contained on the order of a few dozen examples of 8 common classes, as shown in the following table.

|         | Airplane | Bicycle | Boat | Bus | Car | Motorcycle | Train | Truck |
|---------|----------|---------|------|-----|-----|------------|-------|-------|
| # train | 38       | 21      | 20   | 40  | 19  | 30         | 44    | 22    |
| # test  | 60       | 46      | 79   | 69  | 47  | 54         | 67    | 81    |
|         |          |         |      |     |     |            |       |       |

CLIP provides a powerful paradigm for this type of problem. It learns to associate text to images by embedding them in a common space. One can then construct a classifier by providing a set of possible captions that mention the possible labels for an image and select the caption with the highest similarity to the image.

There are several ways to generate captions for each image. One standard way is to use the phrase ‘a sentence of a {object}’ where {object} is replaced by the possible class names. Adding words to disambiguate similar objects sometimes brings about an improvement, as does using multiple prompts and using their mean as an embedding. For simplicity, in this work the simple prompt mentioned above was used.

We can finetune CLIP in several ways: using the entire the network end-to-end or mereley the image encoder only. The latter might make sense in situations where we expect the captions to have been seen and mereley wish to ensure the new images' embeddings are pushed as close as possible to the text embeddings.

The code in this repo was based on https://github.com/openai/CLIP/issues/83.

### Results

Without finetuning CLIP’s top-1 accuracy on the few-shot test data is 89.2%, and top-5 accuracy approaches 100%. This is a formidable baseline, although the latter number is not too surprising given a total of only 8 classes.

The best finetuning performance was 91.3% after 24 epochs of training using a learning rate of 1e-7 and weight decay of 0.0001. Using higher learning rates and a higher weight decay in line with the values mentioned in the paper generally led to slightly worse results. Overall, it was observed that no hyperparameter combination yielded a performance significantly above the baseline, suggesting that CLIP pretraining is quite powerful for the classes in the few-shot dataset provided.

### Future Work

Beyond the scope of this task several methods to improve performance would be worth investigating:

- Adding image augmentations to improve robustness.
- In this work the training and testing data was used as is after a quick look over it suggested no obviously bad data. A natural next step would be to investigate this more closely and ensure there’s no mislabeled examples or ambiguity affecting the performance.
- When deploying the model we can accelerate inference by pre-computing the text embeddings and using those as a linear classification head since the labels we are interested in are known in the few-shot case.
- More complex training regimes using curriculum learning, learning rate schedulers, etc. would be worth trying for further improvement.

### Zero-Shot Learning

The zero shot case described in the task refers to a case where we have neither example images of the data we might want to classify, nor do we know what those examples are.

There are several ways we can approach this problem.

1. We could naively retry the approach described in the Few-Shot Classification section but provide a larger variety of prompts. Assuming the we have trained on a significant share of english language words we may well have encountered the object in the larger pretraining dataset, even if we don’t have explicit examples of it. Ideally, the learned representation is sufficiently powerful such that we hope to discover new object classes with relations such as King - Man + Woman = Queen. Naturally, this approach is limited since some categories may never have been encountered.
2. We could estimate the probability of an image belonging to a new class by estimating how uncertain a prediction is w/r/t the known classes. If e.g. all 8 classes from the few-shot dataset are assigned a probability of 12.5% our prediction will be no better than random guessing. In this case we could use a separate captioning network to produce the most likely caption, add that to the text embeddings and continue using that.
3. We can try to perform clustering in the latent space and whenever an image is too far away from the nearest cluster establish a new cluster center. This is a very general approach but requires a sufficiently good learnt representation.
4. complex combinations of examples - should we separate into disparate classes or not?



## Usage

To finetune, follow the instruction to set up CLIP on the repo's homepage and then run `python few_shot_finetuning.py`.