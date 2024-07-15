import dataset
import detection
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


class CarClassifier_bonus:
    def __init__(self, train_data, test_data):

        self.x_train, self.y_train, self.x_test, self.y_test = None, None, None, None
        self.x_train = np.array([data[0].reshape(-1) for data in train_data])
        self.y_train = np.array([data[1] for data in train_data])
        self.x_test = np.array([data[0].reshape(-1) for data in test_data])
        self.y_test = np.array([data[1] for data in test_data])
        self.model = self.build_model()
        
    
    def build_model(self):
        model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.4, max_depth=3)
        return model

    def train(self):
        trained_model = self.model.fit(self.x_train, self.y_train)
        return trained_model
    
    def eval(self):
        y_pred = self.model.predict(self.x_test)
        print(f"Accuracy: {round(accuracy_score(y_pred, self.y_test), 4)}")
        print("Confusion Matrix: ")
        print(confusion_matrix(y_pred, self.y_test))
    
    def classify(self, input):
        return self.model.predict(input)[0]


# main 
train_data = dataset.load_images('data/train')
test_data = dataset.load_images('data/test')

print("Using GradientBoostingClassifier: \n")
car_clf = CarClassifier_bonus(
    train_data=train_data,
    test_data=test_data
)
car_clf.train()
car_clf.eval()
