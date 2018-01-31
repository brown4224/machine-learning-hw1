#Sean McGlincy
#Homework 1
#KNN

import numpy as np

def knn(training_data, data, k=5):
    #Acumulator
    distance = []

    # Cycle through data and calculate distance  28 x 28 features
    for test_data in training_data:

        #Normalize data here

        dist = 0
        for i in range(1, len(test_data)):
            dist += (test_data[i] - data[i])**2
        dist = np.math.sqrt(dist)


        # Add distance and training value
        item = (dist, int (test_data[0]) )
        distance.append(item)

    #Sort Data
    # np.sort(distance)
    distance.sort(key=lambda tup: tup[0])
    print(distance)

    #Rank K elements
    ranking_array = np.zeros(10, dtype=int)
    for i in range(0, k):

        num = distance[i][1]
        ranking_array[ num ]  += 1

    print(ranking_array)
    return ranking_array.argmax(axis=0)




training_data = np.genfromtxt('MNIST_training.csv', delimiter=',', skip_header=1)
test_data = np.genfromtxt('MNIST_test.csv', delimiter=',', skip_header=1)

k = 5

results =  knn(training_data, test_data[28], k)
print(results)
# print(my_data[0][:-1] )


# items = (1,2)
# items_arr = [items, items]
# print(items_arr)