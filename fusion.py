import numpy as np, cv2, math, time
from keras.layers import Dense, Input
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import label_binarize
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from keras.callbacks import EarlyStopping

class Fusion(object):
    def __init__(self, algorithms, class_names):
        self.algorithms = algorithms
        self.class_names = class_names
        self.number_of_classes = 0
        self.training_data = []
        self.training_labels = []
        
        
    def preprocess_list(self, image_paths):
        images = []
        
        for _class in image_paths:
            images.extend(_class)
        
        return np.array(images)
    
    def ROC_curve(self, y_bin, y_pred_prob, separate_subjects, roc_title):
        plt.figure('ROC curve')
        line_styles = [':', '-', '--', '-.']
        if(separate_subjects):
            for i in range(self.number_of_classes):
                fpr, tpr, _ = roc_curve(y_bin[:, i], y_pred_prob[:, i])
                AUC = auc(fpr, tpr)
                plt.plot(fpr, tpr, lw = 2, color = np.random.rand(3,), linestyle = line_styles[i % len(line_styles)], label = f'ROC curve for {self.class_names[i]} with AUC = {round(AUC, 5)}')
        else:
            fpr, tpr, threshold = roc_curve(y_bin.ravel(), y_pred_prob.ravel())
            AUC = auc(fpr, tpr)
            fnr = 1 - tpr
            index = np.argmin(np.abs((fnr - fpr)))
            EER = np.mean((fnr[index], fpr[index]))
            plt.plot(fpr, tpr, lw = 2, color = 'red', label = f'ROC curve for {self.number_of_classes} individuals with AUC = {round(AUC, 5)}')
            plt.plot(EER, 1 - EER, 'bo', label = f'EER = {EER}')
            plt.plot([1, 0], [0, 1], linestyle = '--')
            
        plt.title(roc_title)
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.legend(loc = 'lower right')

    def FAR_FRR(self, X_test, y_test, model, flip = True):
        genuine_attempts = []
        imposter_attempts = []
        pred_proba = model(X_test)
        
        print('Calculating FAR-FRR plot...')
        for label in tqdm(range(len(np.unique(self.training_labels)))):
            
            y_genuine_probability = pred_proba[y_test == label]
            prediction = np.argmax(y_genuine_probability, 1)
            y_genuine_probability = y_genuine_probability[prediction == label]

            y_imposter_probability = pred_proba[y_test != label][:len(y_genuine_probability)]

            genuine_score = y_genuine_probability[:, label]
            genuine_attempts.extend(genuine_score)

            imposter_score = y_imposter_probability[:, label]
            imposter_attempts.extend(imposter_score)

        genuine_attempts = np.array(genuine_attempts) * 100
        imposter_attempts = np.array(imposter_attempts) * 100
        
        plot_genuine = []
        plot_imposter = []
        end_percentage = 101 # from 0 to 100 inclusive.
        for i in range(end_percentage):
            count_genuine = 0
            count_imposter = 0
            
            for genuine_attempt in genuine_attempts:
                if genuine_attempt <= i:
                    count_genuine += 1
            
            for imposter_attempt in imposter_attempts:
                if imposter_attempt >= i:
                    count_imposter += 1
            
            plot_genuine.append(count_genuine)
            plot_imposter.append(count_imposter)
        
        if flip:
            maximum = np.max((np.max(plot_imposter), np.max(plot_genuine)))
            plot_imposter = maximum - plot_imposter
            plot_genuine = maximum - plot_genuine
        
        #EER calculation 
        index = 0
        value = 0
        distance = 1e9
        for i in range(end_percentage):
            d = abs(plot_imposter[i] - plot_genuine[i])
            
            if d < distance:
                distance = d
                index = i
                value = plot_imposter[index]
            
        print(index, value, sep = ' : ')
        plt.figure('FAR FRR')
        plt.plot(plot_imposter, color = 'red', label = 'FAR')
        plt.plot(plot_genuine, color = 'green', label = 'FRR')
        plt.plot(index, value,'bo', label = f'EER: {index/100}')
        plt.legend(loc = 'best')
        plt.xlabel('Thresholds')
        plt.ylabel('Comparisons')
        plt.xlim([0, 100])


    def genuine_vs_imposter(self, X_test, y_test, model):
        genuine_attempts = []
        imposter_attempts = []
        skip_imposter = False
        pred_proba = model(X_test)
        
        print('calculating the Genuine vs Imposter plot...')
        for label in tqdm(range(len(np.unique(self.training_labels)))):
            genuine_proba = pred_proba[y_test == label]
            
            condition = (y_test != label)
            for i in range(0, label):
                condition = np.bitwise_and(condition, y_test != i)
            imposter_proba = pred_proba[condition, label]
            
            if len(pred_proba) <= 0:
                skip_imposter = True
            
            prediction = np.argmax(genuine_proba, -1)
            actual = y_test[y_test == label]
            genuine_proba = genuine_proba[prediction == actual, label]
            
            for i in range(len(genuine_proba) - 1):
                for j in range(i + 1, len(genuine_proba)):
                    distance = np.sqrt((genuine_proba[i] - genuine_proba[j]) ** 2)
                    genuine_attempts.append(distance)
        
            if not skip_imposter:
                for i in range(len(genuine_proba)):
                    for j in range(len(imposter_proba)):
                        distance = np.sqrt((genuine_proba[i] - imposter_proba[j]) ** 2)
                        imposter_attempts.append(distance)
            
            # genuine_attempts.extend(genuine_proba)
            # imposter_attempts.extend(imposter_proba)
        
        
        genuine_attempts = np.array(genuine_attempts) * 100
        imposter_attempts = np.array(imposter_attempts) * 100
        bins = np.array(list(range(0, 101, 1)))
        
        genuine_hist, _ = np.histogram(genuine_attempts, bins)
        imposter_hist, _ = np.histogram(imposter_attempts, bins)
        
        genuine_hist = np.int64((genuine_hist / np.max(genuine_hist)) * 100)
        imposter_hist = np.int64((imposter_hist / np.max(imposter_hist)) * 100)
        
        plt.figure('Genuine vs Imposter distribution')
        plt.plot(genuine_hist, color = 'green', label = 'Genuine score distribution')
        plt.plot(imposter_hist, color = 'red', label = 'Imposter score distribution')
        plt.ylabel('Distribution')
        plt.xlabel('Scores')
        plt.legend(loc = 'best')
    
    def confusion_matrix(self, y_pred, y_true, labels):
        matrix = confusion_matrix(y_true, y_pred)
        display = ConfusionMatrixDisplay(matrix, display_labels = labels)
        display.plot()
        

class FeatureFusion(Fusion):
    def __init__(self, algorithms, class_names):
        super().__init__(algorithms, class_names) 
    

    def extract_features(self, image_paths, batch_size = 2000, image_size = (128, 128)):
        self.number_of_classes = len(image_paths)
        self.training_labels = []
        for i, _class in enumerate(image_paths):
            self.training_labels.extend([i] * len(_class))
        
        image_paths = self.preprocess_list(image_paths)
        
        self.training_data = []
        index = 0
        current_batch = 1
        total_batches = len(image_paths) / batch_size
        total_batches = math.ceil(total_batches)
        
        print(f'Processing {total_batches} batches...')
        while index < len(image_paths):
            images = []
            print(f'Processing batch {current_batch}/{total_batches}...')
            
            for _ in range(batch_size):
                if index >= len(image_paths):
                    break
                
                img = cv2.resize(cv2.imread(image_paths[index]), image_size)
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
            
        
        
    def train_svm(self, flip = True, test_model = True, separate_subjects = False, roc_title = 'ROC curve'):
        svm = OneVsRestClassifier(SVC(kernel = 'linear', max_iter = 1000000, verbose = True, probability = True), n_jobs = -1)
        
        if test_model:
            X_train, X_test, y_train, y_test = train_test_split(self.training_data, self.training_labels, test_size = 0.25, train_size = 0.75, random_state = 250)
            svm.fit(X_train, y_train)
            y_pred_prob = np.array(svm.decision_function(X_test)) 
            y_pred = np.argmax(y_pred_prob, -1)
            y_bin = label_binarize(y_test, classes = list(range(self.number_of_classes)))

            print('Feature fusion:')                    
            print(classification_report(y_test, y_pred, target_names = self.class_names, labels = np.unique(self.training_labels)))
            self.FAR_FRR(X_test, y_test, svm.predict_proba, flip = flip)
            self.genuine_vs_imposter(X_test, y_test, svm.predict_proba)
            self.confusion_matrix(y_pred, y_test, labels = self.class_names)
            self.ROC_curve(y_bin, y_pred_prob, separate_subjects, roc_title)
            plt.show()
        else:
            svm.fit(self.training_data, self.training_labels)

        return svm


    def train(self, epochs = 100, batch_size = 16, model_layer_sizes = (192, 256, 128), flip = True, patience = 15, test_model = True, separate_subjects = False, roc_title = 'ROC curve'):
        layers = [
            Input(shape = (self.training_data.shape[1]))
        ]
        
        for size in model_layer_sizes:
            layers.append(Dense(size, activation = 'relu'))
        
        layers.append(Dense(self.number_of_classes, activation = 'softmax'))
        
        
        model = Sequential(layers)
        model.compile(Adam(), 'sparse_categorical_crossentropy', ['accuracy'])
        callback = EarlyStopping(patience = patience, verbose = 1, restore_best_weights = True, monitor = 'val_accuracy')
        
        if test_model:
            X_train, X_test, y_train, y_test = train_test_split(self.training_data, self.training_labels, test_size = 0.25, train_size = 0.75, random_state = 250)
            model.fit(X_train, y_train, batch_size = batch_size, epochs = epochs, shuffle = False, use_multiprocessing = True, validation_split = 0.1, callbacks = [callback])
            y_pred_prob = np.array(model.predict(X_test)) 
            y_pred = np.argmax(y_pred_prob, -1)
            y_bin = label_binarize(y_test, classes = list(range(self.number_of_classes)))

            print('Feature fusion:')                    
            print(classification_report(y_test, y_pred, target_names = self.class_names, labels = np.unique(self.training_labels)))
            
            predict = lambda x: model.predict(x, verbose = 0)
            self.FAR_FRR(X_test, y_test, predict, flip)
            self.genuine_vs_imposter(X_test, y_test, predict)
            self.confusion_matrix(y_pred, y_test, labels = self.class_names)
            self.ROC_curve(y_bin, y_pred_prob, separate_subjects, roc_title)
            plt.show()
        else:
            model.fit(self.training_data, self.training_labels, batch_size = batch_size, epochs = epochs, shuffle = False, validation_split = 0.1, callbacks = [callback])
        
        return model


        
        
class ScoreFusion(Fusion):
    def __init__(self, algorithms, class_names, weights = None):
        super().__init__(algorithms, class_names)
        self.models = []
        
        if weights is not None:
            self.weights = weights
        else:
            self.weights = [1 for _ in self.algorithms]  
    
    
    def extract_features(self, image_paths, batch_size = 2000, image_size = (128, 128)):
        self.number_of_classes = len(image_paths)
        
        for i, _class in enumerate(image_paths):
            self.training_labels.extend([i] * len(_class))
            
        image_paths = self.preprocess_list(image_paths)

        self.training_data = [None for _ in self.algorithms]
        index = 0
        current_batch = 1
        total_batches = len(image_paths) / batch_size
        total_batches = math.ceil(total_batches)
        
        print(f'Processing {total_batches} batches...')
        
        while index < len(image_paths):
            images = []
            print(f'Processing batch {current_batch}/{total_batches}...')
            
            for _ in range(batch_size):
                if index >= len(image_paths):
                    break
                
                img = cv2.resize(cv2.imread(image_paths[index]), image_size)
                images.append(img)
                index += 1
            
            images = np.array(images)
            for i, algorithm in enumerate(self.algorithms):
                feature = np.array(algorithm(images))
                
                if self.training_data[i] is None:
                    self.training_data[i] = feature
                else:
                    self.training_data[i] = np.concatenate((self.training_data[i], feature))
                
            current_batch += 1
        
        self.training_labels = np.array(self.training_labels)

        
    def train_svm(self, flip = True, test_model = True, separate_subjects = False, roc_title = 'ROC curve'):
        models = []
        X_tests, y_true = [], None

        for i in range((len(self.algorithms))):
            svm = OneVsRestClassifier(SVC(max_iter = 1000000, verbose = True, probability = True), n_jobs = -1)
            if test_model:
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
        if test_model:
            y_pred_prob = self.vote_svm(X_tests)
            y_bin = label_binarize(y_true, classes = list(range(self.number_of_classes)))
            y_pred = np.argmax(y_pred_prob, -1)
            
            print('Score fusion:')
            print(classification_report(y_true, y_pred, target_names = self.class_names, labels = np.unique(self.training_labels)))
            self.FAR_FRR(X_tests, y_test, self.vote_svm, flip)
            self.genuine_vs_imposter(X_tests, y_test, self.vote_svm)
            self.confusion_matrix(y_pred, y_test, labels = self.class_names)
            self.ROC_curve(y_bin, y_pred_prob, separate_subjects, roc_title)
            plt.show()
        return models
    
    
    def train(self, epochs = 100, batch_size = 16, model_layer_sizes = (192, 256, 128), flip = True, patience = 15, test_model = True, separate_subjects = False, roc_title = 'ROC curve'):
        models = []
        X_tests, y_true = [], None 
        for i in range(len(self.algorithms)):
            layers = [
                Input(shape = (self.training_data[i].shape[1]))
            ]
            
            for size in model_layer_sizes:
                layers.append(Dense(size, 'relu'))
            layers.append(Dense(self.number_of_classes, activation = 'softmax'))
            
            model = Sequential(layers)
            model.compile(Adam(), 'sparse_categorical_crossentropy', ['accuracy'])
            callback = EarlyStopping(patience = patience, verbose = 1, restore_best_weights = True)

            if test_model:
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
        if test_model:
            y_pred_prob = self.vote(X_tests)
            y_bin = label_binarize(y_true, classes = list(range(self.number_of_classes)))
            y_pred = np.argmax(y_pred_prob, -1)
            
            print('Score fusion:')
            print(classification_report(y_true, y_pred, target_names = self.class_names, labels = np.unique(self.training_labels)))
            self.FAR_FRR(X_tests, y_test, self.vote, flip)
            self.genuine_vs_imposter(X_tests, y_test, self.vote)
            self.confusion_matrix(np.argmax(y_pred_prob, -1), y_test, labels = self.class_names)
            self.ROC_curve(y_bin, y_pred_prob, separate_subjects, roc_title)
            plt.show()

        return models


    def vote_svm(self, samples_per_model):
        predictions = None
        for i, model in enumerate(self.models):
            pred = model.predict_proba(samples_per_model[i]) * self.weights[i]

            if predictions is None:
                predictions = pred
            else:
                predictions = np.add(predictions, pred)
        
        return predictions / (sum(self.weights) + 1e-8)
    
    
    def vote(self, samples_per_model):
        predictions = None
        for i, model in enumerate(self.models):
            pred = model.predict(samples_per_model[i], verbose = 0) * self.weights[i]
            if predictions is None:
                predictions = pred
            else:
                predictions = np.add(predictions, pred)
        
        return predictions / (sum(self.weights) + 1e-8)