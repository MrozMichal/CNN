import random
from random import seed
from random import randrange
from random import uniform
from csv import reader
from math import exp
import copy
prev_weights=list()

class NeuralNetwork:
    def __init__(self, activation_ind):
        self.model = models.Sequential()
        self.activations_list = [activations.linear, step_function, activations.sigmoid]
        self.model.add(layers.Dense(1, activation=self.activations_list[activation_ind], input_shape=(2,)))
        self.training_history = None

    def cross_validation_split(self,dataset, n_folds):
        dataset_split = list()
        dataset_copy = list(dataset)
        fold_size = int(len(dataset) / n_folds)

        for i in range(n_folds):
            fold = list()
            while len(fold) < fold_size:
                index = randrange(len(dataset_copy))
                fold.append(dataset_copy.pop(index))
            dataset_split.append(fold)
        return dataset_split


    def accuracy_metric(actual, predicted):
        correct = 0
        for i in range(len(actual)):
            if actual[i] == predicted[i]:
                correct += 1
        return correct / float(len(actual)) * 100.0


    def evaluate_algorithm(self, dataset, algorithm, n_folds, l_rate, n_epoch, network, folds, *args,rand_range,momentum,bias):
        #folds = cross_validation_split(self,dataset, n_folds)
        scores = list()
        print("przed petla for w evaluate")
        for fold in folds:
            train_set = list(folds)
            #Zmienione
            #train_set.remove(fold)
            #
            train_set = sum(train_set, [])
            test_set = list()
            for row in fold:
                row_copy = list(row)
                test_set.append(row_copy)
                row_copy[-1] = None
            predicted = algorithm(self,train_set, test_set, l_rate, n_epoch, network, *args,rand_range,momentum,bias)
            actual = [row[-1] for row in fold]
            accuracy = accuracy_metric(actual, predicted)
            scores.append(accuracy)
        return scores


    def activate(weights, inputs):
        activation = weights[-1]
        for i in range(len(weights) - 1):
            activation += weights[i] * inputs[i]
        return activation


    def transfer(activation):
        return 1.0 / (1.0 + exp(-activation))


    def forward_propagate(network, row):
        inputs = row
        for layer in network:
            new_inputs = []
            for neuron in layer:
                activation = activate(neuron['weights'], inputs)
                neuron['output'] = transfer(activation)
                new_inputs.append(neuron['output'])
            inputs = new_inputs
        return inputs


    def transfer_derivative(output):
        return output * (1.0 - output)


    def backward_propagate_error(network, expected):
        for i in reversed(range(len(network))):
            layer = network[i]
            errors = list()
            if i != len(network) - 1:
                for j in range(len(layer)):
                    error = 0.0
                    for neuron in network[i + 1]:
                        error += (neuron['weights'][j] * neuron['delta'])
                    errors.append(error)
            else:
                for j in range(len(layer)):
                    neuron = layer[j]
                    errors.append(neuron['output'] - expected[j])
            for j in range(len(layer)):
                neuron = layer[j]
                neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])


    def update_weights(network, row, l_rate, momentum,bias):
        for i in range(len(network)):
            #print(i)
            inputs = row[:-1]
            if i != 0:
                inputs = [neuron['output'] for neuron in network[i - 1]]
            for neuron in network[i]:
                #print(inputs)
                for j in range(len(inputs)):
                    # neuron['weights'][j] -= l_rate * neuron['delta'] * inputs[j]
                    buffor = neuron['weights'][j]
                    neuron['weights'][j] -= (l_rate * neuron['delta'] * inputs[j] + momentum * (
                                neuron['weights'][j] - neuron['prev_weights'][j])) + bias
                    neuron['prev_weights'][j] = buffor
                neuron['weights'][-1] -= l_rate * neuron['delta']


    def train_network(network, train, l_rate, n_epoch, n_outputs, rand_range, bias, momentum):
        for epoch in range(n_epoch):
            for row in train:
                outputs = forward_propagate(network, row)
                expected = [0 for i in range(n_outputs)]
                expected[row[-1]] = 1
                backward_propagate_error(network, expected)
                update_weights(network, row, l_rate, momentum,bias)
        for i in range(len(network)):
            for neuron in network[i]:
                print(neuron['weights'])


    def initialize_network(self, train, test, rand_range):
        # Fragment z def.back_propagation
        self.n_inputs = len(train[0]) - 1
        self.n_outputs = len(set([row[-1] for row in train]))
        self.n_hidden = 1
        # Fragment z def.back_propagation

        self.network = list()
        self.Wages = [random.uniform(-rand_range,rand_range) for i in range(self.n_inputs + 1)]
        self.hidden_layer = [{'weights': [random.uniform(-rand_range,rand_range) for i in range(self.n_inputs + 1)],'prev_weights': [0 for i in range(self.n_inputs + 1)]} for i in range(self.n_hidden)]
        self.network.append(self.hidden_layer)
        self.output_layer = [{'weights': [random.uniform(-rand_range,rand_range) for i in range(self.n_hidden + 1)],'prev_weights': [0 for i in range(self.n_hidden + 1)]} for i in range(self.n_outputs)]
        self.network.append(self.output_layer)
        return self.network, self.Wages


    def predict(network, row):
        outputs = forward_propagate(network, row)
        return outputs.index(max(outputs))


    def back_propagation(self,train, test, l_rate, n_epoch, network,train_network, rand_range, bias, momentum):
        #n_inputs = len(train[0]) - 1
        #n_outputs = len(set([row[-1] for row in train]))
        #network = initialize_network(n_inputs, n_hidden, n_outputs)

        #train_network(network, train, l_rate, n_epoch, n_outputs)
        train_network(self.network1, self.train_dataset, self.lear_rate, self.num_epoch, n_outputs, bias, momentum)

        predictions = list()

        for row in test:
            prediction = predict(network, row)
            predictions.append(prediction)
        return (predictions)

    # Test Backprop on Seeds dataset
    #seed(1)
    # load and prepare data
    #   Nalezy zaladowac plik .csv z danymi dla pierwszej sieci neuronowej - pozostale utworza sie same.
    #   Forma pliku wyglada nastepujaco - splaszczona lista, z jedynkami, tam gdzie ma byc True dla ikonki i na końcu stwierdzenie przynaleznosci do klasy, lub jej brak

    """
    # evaluate algorithm
    n_folds = 1
    #l_rate = 0.9
    #n_epoch = 10000
    n_hidden = 1


    scores = evaluate_algorithm(dataset, back_propagation, n_folds, l_rate, n_epoch, n_hidden)
    scores2 = evaluate_algorithm(dataset2, back_propagation, n_folds, l_rate, n_epoch, n_hidden)
    scores3 = evaluate_algorithm(dataset3, back_propagation, n_folds, l_rate, n_epoch, n_hidden)
    """
    # RESULTS

    #print('Scores: %s' % scores)
    #print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))

    #print('Scores: %s' % scores2)
    #print('Mean Accuracy: %.3f%%' % (sum(scores2) / float(len(scores2))))

    #print('Scores: %s' % scores3)
    #print('Mean Accuracy: %.3f%%' % (sum(scores3) / float(len(scores3))))