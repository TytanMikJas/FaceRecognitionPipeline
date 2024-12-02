import matplotlib.pyplot as plt
import itertools
import tensorflow as tf
import numpy as np
import cv2
from typing import cast 

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues): # type: ignore
  if normalize:
      cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
      print("Normalized confusion matrix")
  else:
      print('Confusion matrix, without normalization')

  print(cm)

  plt.imshow(cm, interpolation='nearest', cmap=cmap)
  plt.title(title)
  plt.colorbar()
  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks, classes, rotation=45)
  plt.yticks(tick_marks, classes)

  fmt = '.2f' if normalize else 'd'
  thresh = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
      plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')

IMG_SIZE = (128, 128)
class PhaseTwo:
  def __init__(self, model_path: str):
    input_shape = (*IMG_SIZE, 3)
    model = tf.keras.models.load_model(model_path, custom_objects={'input_shape': input_shape})
    assert model is not None, "Could not import the model"
    self.model = cast(tf.keras.Model, model) 
    
  def predict(self, img: np.ndarray) -> tuple[float, float, float]:
    img = cv2.resize(img, IMG_SIZE)
    img = img / 255.0
    img = img.reshape(1, *IMG_SIZE, 3)
    hand, low_qual, normal = self.model.predict(img)[0]
    return hand, low_qual, normal
  
if __name__ == '__main__':
  phase_two = PhaseTwo('../models/phase_2/Small-Model.keras')
  img = cv2.imread('../assets/model.png') 
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  res = phase_two.predict(img)
  print(res)
  