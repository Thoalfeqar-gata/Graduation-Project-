import numpy as np, cv2, math
from keras.layers import Dense, Input
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from keras.callbacks import EarlyStopping

class FeatureFusion(object):
    def __init__(self, algorithms, class_names, roc_title, model_layer_sizes = (192, 256, 128), test_model = True, batch_size = 1000, image_size = (224, 224), separate_subjects = False, patience = 30):
        self.model_layers_sizes = model_layer_sizes
        self.algorithms = algorithms
        self.separate_subjects = separate_subjects
        self.batch_size = batch_size
        self.image_size = image_size
        self.roc_title = roc_title
        self.patience = patience
        
        self.number_of_classes = 0
        self.training_data = None
        self.training_labels = None
        self.test_model = test_model
        self.class_names = class_names
        
    
    def preprocess_list(self, image_paths):
        images = []
        
        for _class in image_paths:
            images.extend(_class)
        
        return np.array(images)   
    
    
    def extract_features(self, image_paths):
        self.number_of_classes = len(image_paths)
        self.training_labels = []
        for i, _class in enumerate(image_paths):
            self.training_labels.extend([i] * len(_class))
        
        
        image_paths = self.preprocess_list(image_paths)
        
        self.training_data = []
        
        index = 0
        current_batch = 1
        total_batches = len(image_paths) / self.batch_size
        total_batches = math.ceil(total_batches)
        
        print(f'Processing {total_batches} batches...')
        
        while index < len(image_paths):
            images = []
            print(f'Processing batch {current_batch}/{total_batches}...')
            
            for _ in range(self.batch_size):
                if index >= len(image_paths):
                    break
                
                img = cv2.resize(cv2.imread(image_paths[index]), self.image_size)
                images.append(img)
                index += 1
            
            images = np.array(images)
            features = None
            for algorithm in self.algorithms:
                feature = np.array(algorithm(images))
                
                if features is not None:
                    features = np.concatenate((features, feature), axis = 1)
                else:
                    features = feature
            
            self.training_data.extend(features)
            current_batch += 1
        
        self.training_data = np.array(self.training_data)
        self.training_labels = np.array(self.training_labels)    
            
        
        
    def train_svm(self):
        svm = OneVsRestClassifier(SVC(max_iter = 1000000, verbose = True), n_jobs = -1)
        
        if self.test_model:
            X_train, X_test, y_train, y_test = train_test_split(self.training_data, self.training_labels, test_size = 0.25, train_size = 0.75, random_state = 250)
            svm.fit(X_train, y_train)
            y_pred_prob = np.array(svm.decision_function(X_test)) 
            y_bin = label_binarize(y_test, classes = list(range(self.number_of_classes)))

            line_styles = [':', '-', '--', '-.']
            print('Feature fusion:')                    
            print(classification_report(y_test, np.argmax(y_pred_prob, -1), target_names = self.class_names, labels = np.unique(self.training_labels)))
        
            if(self.separate_subjects):
                for i in range(self.number_of_classes):
                    fpr, tpr, _ = roc_curve(y_bin[:, i], y_pred_prob[:, i])
                    AUC = auc(fpr, tpr)
                    plt.plot(fpr, tpr, lw = 2, color = np.random.rand(3,), linestyle = line_styles[i % len(line_styles)], label = f'ROC curve for {self.class_names[i]} with AUC = {round(AUC, 5)}')
            else:
                fpr, tpr, _ = roc_curve(y_bin.ravel(), y_pred_prob.ravel())
                AUC = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, lw = 2, color = 'red', label = f'ROC curve for {self.number_of_classes} individuals with AUC = {round(AUC, 5)}')
            plt.title(self.roc_title)
            plt.ylabel('True Positive Rate')
            plt.xlabel('False Positive Rate')
            plt.legend(loc = 'best')
            plt.show()
        else:
            svm.fit(X_train, y_train)

        
        return svm
    
    
    def train(self, epochs = 100, batch_size = 16):
        layers = [
            Input(shape = (self.training_data.shape[1]))
        ]
        
        for size in self.model_layers_sizes:
            layers.append(Dense(size, activation = 'relu'))
        
        layers.append(Dense(self.number_of_classes, activation = 'softmax'))
        
        
        model = Sequential(layers)
        model.compile(Adam(), 'sparse_categorical_crossentropy', ['accuracy'])
        callback = EarlyStopping(patience = self.patience, verbose = 1, restore_best_weights = True, monitor = 'val_accuracy')
        
        if self.test_model:
            X_train, X_test, y_train, y_test = train_test_split(self.training_data, self.training_labels, test_size = 0.25, train_size = 0.75, random_state = 250)
            model.fit(X_train, y_train, batch_size = batch_size, epochs = epochs, shuffle = False, use_multiprocessing = True, validation_split = 0.1, callbacks = [callback])
            y_pred_prob = np.array(model.predict(X_test)) 
            y_bin = label_binarize(y_test, classes = list(range(self.number_of_classes)))

            line_styles = [':', '-', '--', '-.']
            print('Feature fusion:')                    
            print(classification_report(y_test, np.argmax(y_pred_prob, -1), target_names = self.class_names, labels = np.unique(self.training_labels)))
        
            if(self.separate_subjects):
                for i in range(self.number_of_classes):
                    fpr, tpr, _ = roc_curve(y_bin[:, i], y_pred_prob[:, i])
                    AUC = auc(fpr, tpr)
                    plt.plot(fpr, tpr, lw = 2, color = np.random.rand(3,), linestyle = line_styles[i % len(line_styles)], label = f'ROC curve for {self.class_names[i]} with AUC = {round(AUC, 5)}')
            else:
                fpr, tpr, _ = roc_curve(y_bin.ravel(), y_pred_prob.ravel())
                AUC = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, lw = 2, color = 'red', label = f'ROC curve for {self.number_of_classes} individuals with AUC = {round(AUC, 5)}')
            plt.title(self.roc_title)
            plt.ylabel('True Positive Rate')
            plt.xlabel('False Positive Rate')
            plt.legend(loc = 'best')
            plt.show()
        else:
            model.fit(self.training_data, self.training_labels, batch_size = batch_size, epochs = epochs, shuffle = False, validation_split = 0.1, callbacks = [callback])
        
        return model
        
        
        
class ScoreFusion(object):
    def __init__(self, algorithms, class_names, roc_title, model_layer_sizes = (192, 256, 128), weights = None, batch_size = 1000, image_size = (224, 224),  test_model = True, separate_subjects = False, patience = 30):
        self.model_layer_sizes = model_layer_sizes
        self.algorithms = algorithms
        self.test_model = test_model
        self.class_names = class_names
        self.separate_subjects = separate_subjects
        self.batch_size = batch_size
        self.image_size = image_size
        self.roc_title = roc_title
        self.patience = 30
        
        self.number_of_classes = 0
        self.training_data = []
        self.training_labels = []
        self.models = []
        
        if weights is not None:
            self.weights = weights
        else:
            self.weights = [1 for _ in self.algorithms]  
    

    def preprocess_list(self, images_list):
        images = []
        
        for _class in images_list:
            images.extend(_class)
        
        return np.array(images)
    
    
    def extract_features(self, image_paths):
        self.number_of_classes = len(image_paths)
        
        for i, _class in enumerate(image_paths):
            self.training_labels.extend([i] * len(_class))
            
        image_paths = self.preprocess_list(image_paths)

        self.training_data = [None for _ in self.algorithms]
        index = 0
        current_batch = 1
        total_batches = len(image_paths) / self.batch_size
        total_batches = math.ceil(total_batches)
        
        print(f'Processing {total_batches} batches...')
        
        while index < len(image_paths):
            images = []
            print(f'Processing batch {current_batch}/{total_batches}...')
            
            for _ in range(self.batch_size):
                if index >= len(image_paths):
                    break
                
                img = cv2.resize(cv2.imread(image_paths[index]), self.image_size)
                images.append(img)
                index += 1
            
            images = np.array(images)
            features = None
            for i, algorithm in enumerate(self.algorithms):
                feature = np.array(algorithm(images))
                
                if self.training_data[i] is None:
                    self.training_data[i] = feature
                else:
                    self.training_data[i] = np.concatenate((self.training_data[i], feature))
                
                
            current_batch += 1
        
        self.training_labels = np.array(self.training_labels)

        
    def train_svm(self):
        models = []
        X_tests, y_true = [], None

        for i in range((len(self.algorithms))):
            svm = OneVsRestClassifier(SVC(max_iter = 1000000, verbose = True), n_jobs = -1)
            
            if self.test_model:
                X_train, X_test, y_train, y_test = train_test_split(self.training_data[i], self.training_labels, shuffle = True, random_state = 250, test_size = 0.25, train_size = 0.75)
                X_tests.append(X_test)
                
                if y_true is None:
                    y_true = y_test
                svm.fit(X_train, y_train)
            else:
                svm.fit(self.training_data[i], self.training_labels)

            models.append(svm)
        self.models = models

        np.random.seed(0)
        if self.test_model:
            y_pred_prob = self.vote_svm(X_tests)
            y_bin = label_binarize(y_true, classes = list(range(self.number_of_classes)))
            y_pred = np.argmax(y_pred_prob, -1)
            
            line_styles = [':', '-', '--', '-.']
            print('Score fusion:')
            print(classification_report(y_true, y_pred, target_names = self.class_names, labels = np.unique(self.training_labels)))
            
            if(self.separate_subjects):
                for i in range(self.number_of_classes):
                    fpr, tpr, _ = roc_curve(y_bin[:, i], y_pred_prob[:, i])
                    AUC = auc(fpr, tpr)
                    plt.plot(fpr, tpr, lw = 2, color = np.random.rand(3,), linestyle = line_styles[i % len(line_styles)], label = f'ROC curve for {self.class_names[i]} with AUC = {round(AUC, 5)}')
            else:
                fpr, tpr, _ = roc_curve(y_bin.ravel(), y_pred_prob.ravel())
                AUC = auc(fpr, tpr)
                plt.plot(fpr, tpr, lw = 2, color = 'red', label = f'ROC curve for {self.number_of_classes} individuals with AUC = {round(AUC, 5)}')
            
            plt.ylabel('True Positive Rate')
            plt.xlabel('False Positive Rate')
            plt.legend(loc = 'best')
            plt.show()

        return models
    
    def train(self, epochs = 100, batch_size = 16):
        models = []
        X_tests, y_true = [], None 
        for i in range(len(self.algorithms)):
            layers = [
                Input(shape = (self.training_data[i].shape[1]))
            ]
            
            for size in self.model_layer_sizes:
                layers.append(Dense(size, 'relu'))
            layers.append(Dense(self.number_of_classes, activation = 'softmax'))
            
            model = Sequential(layers)
            model.compile(Adam(), 'sparse_categorical_crossentropy', ['accuracy'])
            callback = EarlyStopping(patience = self.patience, verbose = 1, restore_best_weights = True)

            if self.test_model:
                X_train, X_test, y_train, y_test = train_test_split(self.training_data[i], self.training_labels, test_size = 0.25, train_size = 0.75, random_state = 125, shuffle = True)
                X_tests.append(X_test)
                
                if y_true is None:
                    y_true = y_test
                    
                model.fit(X_train, y_train, batch_size = batch_size, epochs = epochs, shuffle = False, use_multiprocessing = True, callbacks = [callback], validation_split = 0.1)
            else:
                model.fit(self.training_data[i], self.training_labels, batch_size = batch_size, epochs = epochs, shuffle = False, use_multiprocessing = True, callbacks = [callback], validation_split = 0.1)
            
            models.append(model)
        self.models = models
        
        np.random.seed(0)
        if self.test_model:
            y_pred_prob = self.vote(X_tests)
            y_bin = label_binarize(y_true, classes = list(range(self.number_of_classes)))
            y_pred = np.argmax(y_pred_prob, -1)
            
            line_styles = [':', '-', '--', '-.']
            print('Score fusion:')
            print(classification_report(y_true, y_pred, target_names = self.class_names, labels = np.unique(self.training_labels)))
            
            if(self.separate_subjects):
                for i in range(self.number_of_classes):
                    fpr, tpr, _ = roc_curve(y_bin[:, i], y_pred_prob[:, i])
                    AUC = auc(fpr, tpr)
                    plt.plot(fpr, tpr, lw = 2, color = np.random.rand(3,), linestyle = line_styles[i % len(line_styles)], label = f'ROC curve for {self.class_names[i]} with AUC = {round(AUC, 5)}')
            else:
                fpr, tpr, _ = roc_curve(y_bin.ravel(), y_pred_prob.ravel())
                AUC = auc(fpr, tpr)
                plt.plot(fpr, tpr, lw = 2, color = 'red', label = f'ROC curve for {self.number_of_classes} individuals with AUC = {round(AUC, 5)}')
            plt.title(self.roc_title)
            plt.ylabel('True Positive Rate')
            plt.xlabel('False Positive Rate')
            plt.legend(loc = 'best')
            plt.show()

        return models

    def vote_svm(self, samples_per_model):
        predictions = None
        for i, model in enumerate(self.models):
            pred = model.decision_function(samples_per_model[i]) * self.weights[i]

            if predictions is None:
                predictions = pred
            else:
                predictions = np.add(predictions, pred)
        
        return predictions / (sum(self.weights) + 1e-8)
    
    def vote(self, samples_per_model):
        predictions = None
        for i, model in enumerate(self.models):
            pred = model.predict(samples_per_model[i]) * self.weights[i]
            if predictions is None:
                predictions = pred
            else:
                predictions = np.add(predictions, pred)
        
        return predictions / (sum(self.weights) + 1e-8)