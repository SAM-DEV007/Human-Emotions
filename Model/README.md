# Model
  The model uses transfer learning from `ResNetRS50` and the weights of `imagenet`. The implementation of the model is almost 
  same as from my [previous project](https://github.com/SAM-DEV007/MobilePhone-Classifier/blob/main/Model/Model_Training.py). The
  difference is that this model is fine-tuned in its last 84 layers to yield better results.
  
  The loss function used is `SparseCategoricalCrossentropy` with `softmax` activation function in the last layer. The model gives
  out the probabilities of all the seven classes (emotions) and the largest value is chosen. The accuracy obtained is `60%` in
  20 epochs of training.
  
  The [dataset](https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer) contains 28k training images and 7k validation
  images. The image size is `48x48`, and the type is `grayscale`.
