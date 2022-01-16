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

    def cross_validation_split(self):
        dataset_split = list()
        dataset_copy = list(self.train_dataset)
        fold_size = int(len(self.train_dataset) / self.n_folds)

        for i in range(self.n_folds):
            fold = list()
            while len(fold) < fold_size:
                index = randrange(len(dataset_copy))
                fold.append(dataset_copy.pop(index))
            dataset_split.append(fold)
        return dataset_split


    def accuracy_metric(self):
        self.correct = 0
        for i in range(len(self.actual)):
            if self.actual[i] == self.predicted[i]:
                self.correct += 1
        return self.correct / float(len(self.actual)) * 100.0


    def evaluate_algorithm(self):
        self.scores = list()
        for self.fold in self.folds:
            self.train_set = list(self.folds)
            self.train_set = sum(self.train_set, [])
            self.test_set = list()
            for row in self.fold:
                row_copy = list(row)
                self.test_set.append(row_copy)
                row_copy[-1] = None
            self.predicted = back_propagation(self)
            self.actual = [row[-1] for row in fold]
            self.accuracy = accuracy_metric(self)
            self.scores.append(self.accuracy)
        return self.scores


    def activate(self):
        self.activation = self.activate_weights[-1]
        for i in range(len(self.activate_weights) - 1):
            self.activation += self.activate_weights[i] * self.inputs[i]
        return self.activation


    def transfer(self):
        return 1.0 / (1.0 + exp(-self.activation))


    def forward_propagate(self):
        self.inputs = self.row_train
        for self.layer1 in self.network1:
            self.new_inputs1 = []
            for self.neuron1 in self.layer1:
                self.activate_weights = self.neuron1['weights']
                self.activation = activate(self.activate_weights, self.inputs)
                self.neuron1['output'] = transfer(self)
                self.new_inputs1.append(self.neuron1['output'])
            self.inputs1 = self.new_inputs1
        return self.inputs1


    def transfer_derivative(output):
        return output * (1.0 - output)


    def backward_propagate_error(self):
        for i in reversed(range(len(self.network1))):
            layer = self.network1[i]
            errors = list()
            if i != len(self.network1) - 1:
                for j in range(len(layer)):
                    error = 0.0
                    for neuron in self.network1[i + 1]:
                        error += (neuron['weights'][j] * neuron['delta'])
                    errors.append(error)
            else:
                for j in range(len(layer)):
                    neuron = layer[j]
                    errors.append(neuron['output'] - self.expected[j])
            for j in range(len(layer)):
                neuron = layer[j]
                neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])


    def update_weights(self):
        for i in range(len(self.network1)):
            #print(i)
            inputs = self.row_train[:-1]
            if i != 0:
                inputs = [neuron['output'] for neuron in self.network1[i - 1]]
            for neuron in self.network1[i]:
                #print(inputs)
                for j in range(len(inputs)):
                    # neuron['weights'][j] -= l_rate * neuron['delta'] * inputs[j]
                    buffor = neuron['weights'][j]
                    neuron['weights'][j] -= (self.lear_rate * neuron['delta'] * inputs[j] + self.momentum * (
                                neuron['weights'][j] - neuron['prev_weights'][j])) + self.bias
                    neuron['prev_weights'][j] = buffor
                neuron['weights'][-1] -= self.lear_rate * neuron['delta']


    def train_network(self):
        for epoch in range(self.n_epoch):
            for self.row_train in self.train_dataset:
                self.outputs = forward_propagate(self)
                self.expected = [0 for i in range(self.n_outputs)]
                self.expected[row[-1]] = 1
                backward_propagate_error(self)
                update_weights(self)
        for i in range(len(self.network1)):
            for self.neuron in self.network1[i]:
                print(self.neuron['weights'])


    def initialize_network(self):
        # Fragment z def.back_propagation
        self.n_inputs = len(self.train_dataset[0]) - 1
        self.n_outputs = len(set([row[-1] for row in self.train_dataset]))
        self.n_hidden = 1
        # Fragment z def.back_propagation

        self.network = list()
        self.Wages = [random.uniform(-self.random_wages,self.random_wages) for i in range(self.n_inputs + 1)]
        self.hidden_layer = [{'weights': [random.uniform(-self.random_wages,self.random_wages) for i in range(self.n_inputs + 1)],'prev_weights': [0 for i in range(self.n_inputs + 1)]} for i in range(self.n_hidden)]
        self.network.append(self.hidden_layer)
        self.output_layer = [{'weights': [random.uniform(-self.random_wages,self.random_wages) for i in range(self.n_hidden + 1)],'prev_weights': [0 for i in range(self.n_hidden + 1)]} for i in range(self.n_outputs)]
        self.network.append(self.output_layer)
        return self.network, self.Wages


    def predict(self):
        self.outputs = forward_propagate(self)
        return self.outputs.index(max(outputs))


    def back_propagation(self):
        #n_inputs = len(train[0]) - 1
        #n_outputs = len(set([row[-1] for row in train]))
        #network = initialize_network(n_inputs, n_hidden, n_outputs)

        #train_network(network, train, l_rate, n_epoch, n_outputs)
        train_network(self)
        self.predictions = list()

        for self.row_BP in self.test_dataset:
            self.prediction = NN.predict(self.network1, self.row_BP)
            self.predictions.append(self.prediction)
        return (self.predictions)

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