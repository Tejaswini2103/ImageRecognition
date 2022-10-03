import pandas as pandas
import numpy as numpy
import sys
import math
import random
import warnings

class NeuralNetwork:
    warnings.filterwarnings('ignore')

    def __init__(self,w1,w2,w3,b1,b2,b3):
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.b1 = b1
        self.b2 = b2
        self.b3 = b3

    def predict(self, test_images):
        test_images = test_images.T
        test_images = test_images.values
        result = dict()

        tempw1 = numpy.matmul(self.w1, test_images)
        result['wt1'] = tempw1 + self.b1
        result['h1'] = 1.0 / (1.0 + numpy.exp(-result['wt1']))

        tempw2 = numpy.matmul(self.w2, result['h1'])
        result['wt2'] = tempw2 + self.b2
        result['h2'] = 1.0 / (1.0 + numpy.exp(-result['wt2']))

        tempw3 = numpy.matmul(self.w3, result['h2'])
        result['wt3'] = tempw3 + self.b3
        sm = numpy.exp(result['wt3'] - result['wt3'].max())
        result['output'] = sm / numpy.sum(sm, axis=0)

        return result['output']

    def trainNetwork(self, trainingData_images, trainingData_labels):

        batch_size = 32
        l_rate = 0.01

        trainingData_images_T = trainingData_images.T
        trainingData_labels_T = trainingData_labels.T
        trainingData_images_Arr = trainingData_images_T.values
        trainingData_labels_Arr = trainingData_labels_T.values

        trainingLabels_encoded = numpy.zeros((trainingData_labels_Arr.size, 10))
        trainingLabels_encoded[numpy.arange(trainingData_labels_Arr.size), trainingData_labels_Arr] = 1
        trainingLabels_encoded = trainingLabels_encoded.T

        epoch_count = 80
        for i in range(0, epoch_count):
            random = numpy.arange(len(trainingData_images_Arr[1]))
            numpy.random.shuffle(random)
            images = trainingData_images_Arr[:, random]
            labels = trainingLabels_encoded[:, random]

            imagesCount = images[1].size
            all_batches = list()
            all_images = list()
            all_labels = list()
            count = imagesCount / batch_size
            batches_count = math.floor(count)
            n = batch_size

            for k in range(0, batches_count):
                column_start_range = k * n
                column_end_range = (k + 1) * n
                imagesSubset = images[:, column_start_range: column_end_range]
                labelsSubset = labels[:, column_start_range: column_end_range]
                all_images.append(imagesSubset)
                all_labels.append(labelsSubset)

            if imagesCount % n != 0:
                column_start_range_last = n * batches_count
                column_end_range_last = imagesCount
                images_lastSet = images[:, column_start_range_last: column_end_range_last]
                labels_lastSet = labels[:, column_start_range_last: column_end_range_last]
                all_images.append(images_lastSet)
                all_labels.append(labels_lastSet)

            for l in range(len(all_images)):
                all_batches.append((all_images[l], all_labels[l]))

            for batch in all_batches:
                batch_images = batch[0]
                batch_labels = batch[1]

                nodes = dict()

                tempw1 = numpy.matmul(self.w1, batch_images)
                nodes['wt1'] = tempw1 + self.b1
                nodes['h1'] = 1.0 / (1.0 + numpy.exp(-nodes['wt1']))

                tempw2 = numpy.matmul(self.w2, nodes['h1'])
                nodes['wt2'] = tempw2 + self.b2
                nodes['h2'] = 1.0 / (1.0 + numpy.exp(-nodes['wt2']))

                tempw3 = numpy.matmul(self.w3, nodes['h2'])
                nodes['wt3'] = tempw3 + self.b3
                smax = numpy.exp(nodes['wt3'] - nodes['wt3'].max())
                nodes['output'] = smax / numpy.sum(smax, axis=0)

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
                self.w2 = self.w2 - (adjustments['w2_delta'] * l_rate)
                self.w3 = self.w3 - (adjustments['w3_delta'] * l_rate)
                self.b1 = self.b1 - (adjustments['b1_delta'] * l_rate)
                self.b2 = self.b2 - (adjustments['b2_delta'] * l_rate)
                self.b3 = self.b3 - (adjustments['b3_delta'] * l_rate)

            outputreturnedByCode = []
            outputTemp = self.predict(trainingData_images)
            opt = numpy.argmax(outputTemp, axis=0)
            outputreturnedByCode.append(opt == numpy.argmax(trainingLabels_encoded, axis=0))
            print(f'accuracy for epoch {i + 1}: {numpy.mean(outputreturnedByCode)}')


w1factor = numpy.sqrt(1.0 / 784)
w2factor = numpy.sqrt(1.0 / 550)
w3factor = numpy.sqrt(1.0 / 300)

NeuralNetwork = NeuralNetwork(w1=numpy.random.randn(550, 784) * w1factor, w2=numpy.random.randn(300, 550) * w2factor, w3=numpy.random.randn(10, 300) * w3factor,
                              b1=numpy.random.randn(550, 1) * w1factor, b2=numpy.random.randn(300, 1) * w2factor, b3=numpy.random.randn(10, 1) * w3factor)

trainingData_images = pandas.read_csv(sys.argv[1], header=None)
trainingData_labels = pandas.read_csv(sys.argv[2], header=None)
testingData_images = pandas.read_csv(sys.argv[3], header=None)

NeuralNetwork.trainNetwork(trainingData_images, trainingData_labels)
output = NeuralNetwork.predict(testingData_images)

resultOutput = pandas.DataFrame(numpy.argmax(output, axis=0))
resultOutput.to_csv('test_predictions.csv', header=None, index=None)
