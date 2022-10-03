import pandas as pandas
import numpy as numpy
import sys
import math
import random

class NeuralNetwork:

    def __init__(self,w1,w2,w3,b1,b2,b3):
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.b1 = b1
        self.b2 = b2
        self.b3 = b3

    def sigmoid(self, x):
        x = numpy.clip(x, -500, 500)
        return 1.0 / (1.0 + numpy.exp(-x))

    def sigmoid_backward(self, dA, Z):
        sig = self.sigmoid(Z)
        return dA * sig * (1 - sig)

    def softmax(self, x):
        exps = numpy.exp(x - x.max())
        return exps / numpy.sum(exps, axis=0)

    def forward_pass(self, testX_in):
        testX = testX_in.T
        testX = testX.values

        cache = dict()

        cache['Z1'] = numpy.dot(self.w1, testX) + self.b1
        cache['A1'] = self.sigmoid(cache['Z1'])
        cache['Z2'] = numpy.dot(self.w2, cache['A1']) + self.b2
        cache['A2'] = self.sigmoid(cache['Z2'])
        cache['Z3'] = numpy.dot(self.w3, cache['A2']) + self.b3
        cache['A3'] = self.softmax(cache['Z3'])

        return cache

    def trainNetwork(self, trainingData_images, trainingData_labels):

        batch_size = 32
        l_rate = 0.01
        num_epochs = 80

        trainingData_images_T = trainingData_images.T
        trainingData_labels_T = trainingData_labels.T

        trainingData_images_Arr = trainingData_images_T.values
        trainingData_labels_Arr = trainingData_labels_T.values

        trainingLabels_encoded = numpy.zeros((trainingData_labels_Arr.size, trainingData_labels_Arr.max() + 1))
        trainingLabels_encoded[numpy.arange(trainingData_labels_Arr.size), trainingData_labels_Arr] = 1
        trainingLabels_encoded = trainingLabels_encoded.T

        for i in range(0, num_epochs):
            random = numpy.arange(len(trainingData_images_Arr[1]))
            numpy.random.shuffle(random)
            images = trainingData_images_Arr[:, random]
            labels = trainingLabels_encoded[:, random]

            imagesCount = images[1].size
            all_batches = list()
            count = imagesCount / batch_size
            batches_count = math.floor(count)
            n = batch_size

            for i in range(0, batches_count):
                imagesSubset = images[:, i * n: (i + 1) * n]
                labelsSubset = labels[:, i * n: (i + 1) * n]
                all_batches.append((imagesSubset, labelsSubset))

            if imagesCount % n != 0:
                images_lastSet = images[:, n * batches_count: imagesCount]
                labels_lastSet = labels[:, n * batches_count: imagesCount]
                all_batches.append((images_lastSet, labels_lastSet))

            for batch in all_batches:
                batch_images, batch_labels = batch

                nodes = dict()

                tempw1 = numpy.matmul(self.w1, batch_images)
                nodes['wt1'] = tempw1 + self.b1
                nodes['wt1'] = numpy.clip(nodes['wt1'], -500, 500)
                nodes['h1'] = 1.0 / (1.0 + numpy.exp(-nodes['wt1']))

                tempw2 = numpy.matmul(self.w2, nodes['h1'])
                nodes['wt2'] = tempw2 + self.b2
                nodes['wt2'] = numpy.clip(nodes['wt2'], -500, 500)
                nodes['h2'] = 1.0 / (1.0 + numpy.exp(-nodes['wt2']))

                tempw3 = numpy.matmul(self.w3, nodes['h2'])
                nodes['wt3'] = tempw3 + self.b3
                nodes['output'] = self.softmax(nodes['wt3'])

                error = nodes['output'] - batch_labels

                deltaW3 = numpy.matmul(error, nodes["h2"].T)
                deltab3 = numpy.sum(error, keepdims=True, axis=1)

                deltaiH2 = numpy.matmul(self.w3.T, error)
                sigmoidiH2 = 1.0 / (1.0 + numpy.exp(-nodes['wt2']))
                deltaiW2 = deltaiH2 * sigmoidiH2 * (1 - sigmoidiH2)

                deltaW2 = numpy.matmul(deltaiW2, nodes['h1'].T)
                deltab2 = numpy.sum(deltaiW2, keepdims=True, axis=1)

                deltaiH1 = numpy.matmul(self.w2.T, deltaiW2)
                sigmoidiH1 = 1.0 / (1.0 + numpy.exp(-nodes['wt1']))
                deltaiW1 = deltaiH1 * sigmoidiH1 * (1 - sigmoidiH1)

                deltaW1 = numpy.matmul(deltaiW1, batch_images.T)
                deltab1 = numpy.sum(deltaiW1, keepdims=True, axis=1)

                size = batch_images.shape[1]

                adjustments = dict()
                adjustments['w3_delta'] = deltaW3 / size
                adjustments['b3_delta'] = deltab3 / size
                adjustments['w2_delta'] = deltaW2 / size
                adjustments['b2_delta'] = deltab2 / size
                adjustments['w1_delta'] = deltaW1 / size
                adjustments['b1_delta'] = deltab1 / size

                self.w1 = self.w1 - (adjustments['w1_delta'] * l_rate)
                self.b1 = self.b1 - (adjustments['b1_delta'] * l_rate)
                self.w2 = self.w2 - (adjustments['w2_delta'] * l_rate)
                self.b2 = self.b2 - (adjustments['b2_delta'] * l_rate)
                self.w3 = self.w3 - (adjustments['w3_delta'] * l_rate)
                self.b3 = self.b3 - (adjustments['b3_delta'] * l_rate)


NeuralNetwork = NeuralNetwork(w1=numpy.random.randn(512, 784) * numpy.sqrt(1.0 / 784),
                              w2=numpy.random.randn(256, 512) * numpy.sqrt(1.0 / 512), w3=numpy.random.randn(10, 256) * numpy.sqrt(1.0 / 256),
                              b1=numpy.random.randn(512, 1) * numpy.sqrt(1.0 / 748), b2=numpy.random.randn(256, 1) * numpy.sqrt(1.0 / 512),
                              b3=numpy.random.randn(10, 1) * numpy.sqrt(1.0 / 256))

trainingData_images = pandas.read_csv(sys.argv[1], header=None)
trainingData_labels = pandas.read_csv(sys.argv[2], header=None)
testingData_images = pandas.read_csv(sys.argv[3], header=None)

NeuralNetwork.trainNetwork(trainingData_images, trainingData_labels)

cache = NeuralNetwork.forward_pass(testingData_images)

output = cache['A3']
pred = numpy.argmax(output, axis=0)

pandas.DataFrame(pred).to_csv('test_predictions.csv', header=None, index=None)
