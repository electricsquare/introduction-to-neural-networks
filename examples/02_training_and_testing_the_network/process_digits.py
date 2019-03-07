import mnist
from sklearn.neural_network import MLPClassifier

training_images, training_labels, test_images, test_labels = mnist.load()

classifier = MLPClassifier(
  solver='sgd', # Use stochastic gradient descent solver
  activation="logistic", # Use the sigmoid activation function
  hidden_layer_sizes=(14, 14), # Set the hidden layer sizes 
  learning_rate='constant', # Set a constant learning rate
  learning_rate_init=0.01,  # Initialize the learning rate
  max_iter=100
)

classifier.fit(training_images, training_labels)

predicted_labels = classifier.predict(test_images)

num_correct = 0
for predicted_label, test_label in zip(predicted_labels, test_labels):

  if predicted_label == test_label:
    num_correct = num_correct + 1

score = num_correct/len(predicted_labels)

print("We have a score of {}".format(score))