import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

class CarClassifier:
    def __init__(self, model_name, train_data, test_data):

        '''
        Convert the 'train_data' and 'test_data' into the format
        that can be used by scikit-learn models, and assign training images
        to self.x_train, training labels to self.y_train, testing images
        to self.x_test, and testing labels to self.y_test.These four 
        attributes will be used in 'train' method and 'eval' method.
        '''

        self.x_train, self.y_train, self.x_test, self.y_test = None, None, None, None

        # Begin your code (Part 2-1)
        self.x_train = np.array([data[0].reshape(-1) for data in train_data])
        self.y_train = np.array([data[1] for data in train_data])
        self.x_test = np.array([data[0].reshape(-1) for data in test_data])
        self.y_test = np.array([data[1] for data in test_data])
        #raise NotImplementedError("To be implemented")
        # End your code (Part 2-1)
        
        self.model = self.build_model(model_name)
        
    
    def build_model(self, model_name):
        '''
        According to the 'model_name', you have to build and return the
        correct model.
        '''
        # Begin your code (Part 2-2)
        if model_name == "KNN":
            n_neighbors = 1
            weights = 'uniform' 
            p = 1
            #print("\nn_neighbors =" , n_neighbors , "\nweights =" , weights + "\np =" , p , "\n")
            model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, p=p)
        elif model_name == "RF":
            n_estimators = 300
            criterion = 'entropy'
            max_features = 'log2'
            max_samples = 0.8
            #print("\nn_estimators =" , n_estimators , "\ncriterion =" , criterion , "\nmax_features =" , 
            #      max_features , "\nmax_samples =", max_samples ,"\n")
            model = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion , max_features=max_features, max_samples=max_samples)
        else: # "AB"
            estimator = RandomForestClassifier(n_estimators=300, criterion='log_loss' , max_features='log2', max_samples=0.8)
            n_estimators = 500
            learning_rate = 0.4
            #print("\nn_estimators =",n_estimators , "\nlearning_rate =" , learning_rate , "\n")
            model = AdaBoostClassifier(estimator=estimator, n_estimators=n_estimators, learning_rate=learning_rate, algorithm='SAMME')

        return model
        #raise NotImplementedError("To be implemented")
        # End your code (Part 2-2)

    def train(self):
        '''
        Fit the model on training data (self.x_train and self.y_train).
        '''
        # Begin your code (Part 2-3)
        trained_model = self.model.fit(self.x_train, self.y_train)
        return trained_model
        #raise NotImplementedError("To be implemented")
        # End your code (Part 2-3)
    
    def eval(self):
        y_pred = self.model.predict(self.x_test)
        print(f"Accuracy: {round(accuracy_score(y_pred, self.y_test), 4)}")
        print("Confusion Matrix: ")
        print(confusion_matrix(y_pred, self.y_test))
    
    def classify(self, input):
        return self.model.predict(input)[0]
        

