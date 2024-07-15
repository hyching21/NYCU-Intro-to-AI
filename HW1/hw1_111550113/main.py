import dataset
import model
import detection
import matplotlib.pyplot as plt

# Part 1: Implement loadImages function in dataset.py and test the following code.
print('Loading images')
train_data = dataset.load_images('data/train')
print(f'The number of training samples loaded: {len(train_data)}')
test_data = dataset.load_images('data/test')
print(f'The number of test samples loaded: {len(test_data)}')

print('Show the first and last images of training dataset')
fig, ax = plt.subplots(1, 2)
ax[0].axis('off')
ax[0].set_title('Car')
ax[0].imshow(train_data[1][0], cmap='gray')
ax[1].axis('off')
ax[1].set_title('Non car')
ax[1].imshow(train_data[-1][0], cmap='gray')
plt.show()

# Part 2: Build and train 3 kinds of classifiers: KNN, Random Forest and Adaboost.
# Part 3: Modify difference values at parameter n_neighbors of KNeighborsClassifier, n_estimators 
# of RandomForestClassifier and AdaBoostClassifier, and find better results.
car_clf = model.CarClassifier(
    model_name="AB", # KNN, RF (Random Forest) or AB (AdaBoost)
    train_data=train_data,
    test_data=test_data
)
car_clf.train()
car_clf.eval()

# Part 4: Implement detect function in detection.py and test the following code.
print('\nUse your classifier with video.gif to get the predictions (one .txt and one .png)')
detection.detect('data/detect/detectData.txt', car_clf)


# Part 5: Draw line graph
# open txt file and store value
groundtruth = [[0 for _ in range(76)] for _ in range(51)]
knn_pred, rf_pred, ab_pred = [0]*51, [0]*51, [0]*51

with open('GroundTruth.txt', 'r') as file:
        ground_count = [0]*51
        for line_num, line  in enumerate(file.readlines(),1):
            values = line.strip().split()
            for i, value in enumerate(values,0):
                groundtruth[line_num][i] = value
                if value == '1':
                    ground_count[line_num]+=1
 

with open('KNN.txt', 'r') as file:
        knn_count = [0]*51
        for line_num, line  in enumerate(file.readlines(),1):
            values = line.strip().split()
            for i, value in enumerate(values,0):
                if value == '1':
                    knn_count[line_num]+=1
                if value == groundtruth[line_num][i]:
                    knn_pred[line_num] +=1

with open('RF.txt', 'r') as file:
        rf_count = [0]*51
        for line_num, line  in enumerate(file.readlines(),1):
            values = line.strip().split()
            for i, value in enumerate(values,0):
                if value == '1':
                    rf_count[line_num]+=1
                if value == groundtruth[line_num][i]:
                    rf_pred[line_num] +=1

with open('AB.txt', 'r') as file:
        ab_count = [0]*51
        for line_num, line  in enumerate(file.readlines(),1):
            values = line.strip().split()
            for i, value in enumerate(values,0):
                if value == '1':
                    ab_count[line_num]+=1
                if value == groundtruth[line_num][i]:
                    ab_pred[line_num] +=1

# 1. Parking Slots Occupation
                
x_values = list(range(1, 51))
y_values_1 = ground_count[1:51]
y_values_2 = knn_count[1:51]
y_values_3 = rf_count[1:51]
y_values_4 = ab_count[1:51]

plt.plot(x_values, y_values_1, label='Ground Truth')
plt.plot(x_values, y_values_2, label='KNN')
plt.plot(x_values, y_values_3, label='Random Forest')
plt.plot(x_values, y_values_4, label='AdaBoost')

plt.xlabel("Time Slot")
plt.ylabel("#cars")
plt.title("Parking Slots Occupation")

plt.legend()
plt.savefig("Occupation.png")
plt.close()

# 2. Accuracy of the models

x = list(range(1, 51))
for i in range(1,51):
     knn_pred[i] = knn_pred[i]/76
     rf_pred[i] = rf_pred[i]/76
     ab_pred[i] = ab_pred[i]/76
   
plt.plot(x, knn_pred[1:51], label='KNN')
plt.plot(x, rf_pred[1:51], label='Random Forest')
plt.plot(x, ab_pred[1:51], label='AdaBoost')

plt.xlabel("Time Slot")
plt.ylabel("Accuracy")
plt.title("Accuracy of the Models")
plt.legend()
plt.savefig("Accuracy.png")
plt.close()