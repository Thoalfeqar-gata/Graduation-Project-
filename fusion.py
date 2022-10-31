import numpy as np, cv2
from keras.layers import Dense, Input
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
from matplotlib import pyplot as plt

class FeatureFusion(object):
    def __init__(self, algorithms, class_names, model_layer_sizes = (192, 256, 128), test_model = True):
        self.model_layers_sizes = model_layer_sizes
        self.algorithms = algorithms
        
        
        self.number_of_classes = 0
        self.training_data = None
        self.training_labels = None
        self.test_model = test_model
        self.class_names = class_names
    
    def _extract(self, images):
        for algorithm in self.algorithms:
            if self.training_data is None:
                self.training_data = np.array(algorithm(images)).reshape(len(images), -1)
            else:
                self.training_data = np.concatenate((self.training_data, np.array(algorithm(images)).reshape(len(images), -1)), axis = 1)
            
        
    
    def preprocess_list(self, images_list):
        images = []
        
        for _class in images_list:
            images.extend(_class)
        
        return np.array(images)   
    
    
    def extract_features(self, images_list):
        self.number_of_classes = len(images_list)
        
        self.training_labels = []
        for i, _class in enumerate(images_list):
            self.training_labels.extend([i] * len(_class))
        
        
        images_list = self.preprocess_list(images_list)
        self._extract(images_list)
        self.training_labels = np.array(self.training_labels)
        
                
    def train(self, epochs = 250):
        layers = [
            Input(shape = (self.training_data.shape[1]))
        ]
        
        for size in self.model_layers_sizes:
            layers.append(Dense(size, activation = 'relu'))
        
        layers.append(Dense(self.number_of_classes, activation = 'softmax'))
        
        model = Sequential(layers)
        model.compile(Adam(), 'sparse_categorical_crossentropy', ['accuracy'])
        
        if self.test_model:
            X_train, X_test, y_train, y_test = train_test_split(self.training_data, self.training_labels, test_size = 0.25, train_size = 0.75, random_state = 250)
            model.fit(X_train, y_train, epochs = epochs, shuffle = False, use_multiprocessing = True)
            y_pred_prob = model.predict(X_test) 
            y_bin = label_binarize(y_test, classes = list(range(self.number_of_classes)))

            line_styles = [':', '-', '--', '-.']
            print('Feature fusion:')         
            print(classification_report(y_test, np.argmax(y_pred_prob, -1), target_names = self.class_names))

            for i in range(self.number_of_classes):
                fpr, tpr, _ = roc_curve(y_bin[:, i], y_pred_prob[:, i])
                AUC = auc(fpr, tpr)
                plt.plot(fpr, tpr, lw = 2, color = np.random.rand(3,), linestyle = line_styles[i % len(line_styles)], label = f'ROC curve for {self.class_names[i]} with AUC = {round(AUC, 5)}')
            
            plt.ylabel('True Positive Rate')
            plt.xlabel('False Positive Rate')
            plt.legend(loc = 'best')
            plt.show()
        else:
            model.fit(self.training_data, self.training_labels, epochs = epochs, shuffle = False)
        
        return model
        
        
        
class ScoreFusion(object):
    def __init__(self, algorithms, class_names, model_layer_sizes = (192, 256, 128), weights = None,  test_model = True):
        self.model_layer_sizes = model_layer_sizes
        self.algorithms = algorithms
        self.test_model = test_model
        self.class_names = class_names
        self.number_of_classes = 0
        self.training_data = []
        self.training_labels = []
        self.models = []
        
        if weights is not None:
            self.weights = weights
        else:
            self.weights = [1 for _ in self.algorithms.keys()]
    
    def _extract(self, images):
        for algorithm in self.algorithms:
            self.training_data.append(np.array(algorithm(images)).reshape(len(images), -1))
        
    
    def preprocess_list(self, images_list):
        images = []
        
        for _class in images_list:
            images.extend(_class)
        
        return np.array(images)
    
    
    def extract_features(self, images_list):
        self.number_of_classes = len(images_list)
        
        for i, _class in enumerate(images_list):
            self.training_labels.extend([i] * len(_class))
            
        images_list = self.preprocess_list(images_list)
        
        self._extract(images_list)
        self.training_labels = np.array(self.training_labels)
        
        
        
    
    
    def train(self, epochs = 250):
        models = []
        X_tests, y_true = [], None 
        for i in range(len(self.algorithms)):
            layers = [
                Input(shape = (self.training_data[i].shape[1]))
            ]
            
            for size in self.model_layer_sizes:
                layers.append(Dense(size, 'relu'))
            layers.append(Dense(self.number_of_classes, activation = 'softmax'))
            
            model = Sequential(layers, name = list(self.algorithms.keys())[i])
            model.compile(Adam(), 'sparse_categorical_crossentropy', ['accuracy'])
            
            if self.test_model:
                X_train, X_test, y_train, y_test = train_test_split(self.training_data[i], self.training_labels, test_size = 0.25, train_size = 0.75, random_state = 125, shuffle = True)
                X_tests.append(X_test)
                
                if y_true is None:
                    y_true = y_test
                    
                model.fit(X_train, y_train, epochs = epochs, shuffle = False, use_multiprocessing = True)
            else:
                model.fit(self.training_data[i], self.training_labels, epochs = epochs, shuffle = False, use_multiprocessing = True)
            
            models.append(model)
        self.models = models
        
        np.random.seed(0)
        if self.test_model:
            y_pred_prob = self.vote(X_tests)
            y_bin = label_binarize(y_true, classes = list(range(self.number_of_classes)))
            y_pred = np.argmax(y_pred_prob, -1)
            
            line_styles = [':', '-', '--', '-.']
            print('Score fusion:')
            print(classification_report(y_true, y_pred, target_names = self.class_names))
            
            for i in range(len(self.class_names)):
                fpr, tpr, _ = roc_curve(y_bin[:, i], y_pred_prob[:, i])
                AUC = auc(fpr, tpr)
                
                plt.plot(fpr, tpr, lw = 2, color = np.random.rand(3), linestyle = line_styles[i % len(line_styles)], label = f'ROC curve for {self.class_names[i]} with AUC = {round(AUC, 5)}')
            
            plt.ylabel('True Positive Rate')
            plt.xlabel('False Positive Rate')
            plt.legend(loc = 'best')
            plt.show()
        else:
            model.fit(self.training_data, self.training_labels, epochs = epochs, shuffle = False)

        return models

            
    
    
    def vote(self, samples_per_model):
        predictions = None
        for i, model in enumerate(self.models):
            pred = model.predict(samples_per_model[i]) * self.weights[i]
            if predictions is None:
                predictions = pred
            else:
                predictions = np.add(predictions, pred)
        
        return predictions / (sum(self.weights) + 1e-8)
        
        
        