import numpy as np
import math
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

lambdaFactor = 0.1
learningRate = 0.01

print("UBitName = karanhor")
print("personNumber = 50249274")

print("UBitName = pbisht2")
print("personNumber = 50247429")

print('UBitName = rakeshsi')
print('personNumber = 50249135')

def ClusterIndicesNumpy(clustNum, labels):  # numpy
    return np.where(labels == clustNum)[0]


# Closed form solution function
def closed_form_sol(L2_lambda, design_matrix, output_data):
    return np.linalg.solve(
        L2_lambda * np.identity(design_matrix.shape[1]) +
        np.matmul(design_matrix.T, design_matrix),
        np.matmul(design_matrix.T, output_data)
    ).flatten()


# design matrix computation
def compute_design_matrix(X, centers, spreads):
    # use broadcast
    basis_func_outputs = np.exp(
        np.sum(
            np.matmul(X - centers, spreads) * (X - centers), axis=2) / (-2)).T
    # insert ones to the 1st col
    return np.insert(basis_func_outputs, 0, 1, axis=1)


''' Utility Function to Partition and Randomizing Data
    Input : Accepts input_data - Data to be partition
            trainPerc - percentage of the training data
            validPerc - percentage of the validation data
    return : Three arrays containing the training, validation 
            and test data.
'''

def partitionData(input_data, trainPerc, validPerc):
    input_train_div = math.ceil(trainPerc * len(input_data))
    input_val_div = math.ceil((validPerc + trainPerc) * len(input_data))
    np.random.seed(500)
    indices = np.random.permutation(input_data.shape[0])
    training_idx, validate_idx, test_idx = indices[:input_train_div], \
                                           indices[input_train_div:input_val_div], \
                                           indices[input_val_div:]
    input_data_train, input_data_validate, input_data_test = input_data[training_idx, :], \
                                                             input_data[validate_idx, :], \
                                                             input_data[test_idx, :]
    return input_data_train, input_data_validate, input_data_test


'''
    SGD_Sol : Function is used to calculate the 
              SGD Weights using the Early stop mechanism,
			  with patience value 10.
'''

def SGD_sol(learning_rate, minibatch_size, num_epochs, L2_lambda, design_matrix, output_data, no_of_clusters,
            validData,
            designMatrixValidate):
    N, _ = design_matrix.shape
    # You can try different mini-batch size size
    # Using minibatch_size = N is equivalent to standard gradient descent
    # Using minibatch_size = 1 is equivalent to stochastic gradient descent
    # In this case, minibatch_size = N is better
    weights = np.zeros([1, no_of_clusters + 1])
    p = 10;
    lcurr = lprev = 0
    # The more epochs the higher training accuracy. When set to 1000000,
    # weights will be very close to closed_form_weights. But this is unnecessary
    for epoch in range(num_epochs):
        for i in range(N // minibatch_size):
            lower_bound = i * minibatch_size
            upper_bound = min((i + 1) * minibatch_size, N)
            Phi = design_matrix[lower_bound: upper_bound, :]
            t = output_data[lower_bound: upper_bound, :]
            E_D = np.matmul(
                (np.matmul(Phi, weights.T) - t).T,
                Phi
            )
            E = (E_D + L2_lambda * weights) / minibatch_size
            weights = weights - learning_rate * E
        np.linalg.norm(E)
        # calculation for the early stop with step size = 5 epochs
        # and patience factor 10
        if (epoch % 5 == 0):
            lcurr = errorFunc(validData, weights.flatten(), designMatrixValidate, lambdaFactor)
            if (lprev <= lcurr):
                lprev = lcurr
                p = p - 1
                if (p <= 0):
                    print("Early stopping, epoch =  ", epoch)
                    break;
            else:
                lprev = lcurr
                weightsOptimal = weights
                p = 10;

    return weightsOptimal.flatten()


# Loading LETOR Data
letor_input_data = np.genfromtxt('datafiles/Querylevelnorm_X.csv', delimiter=',')
letor_output_data = np.genfromtxt('datafiles/Querylevelnorm_t.csv', delimiter=',').reshape([-1, 1])

letor_input_data_train, letor_input_data_validate, letor_input_data_test = partitionData(letor_input_data, 0.8,
                                                                                         lambdaFactor)
letor_output_data_train, letor_output_data_validate, letor_output_data_test = partitionData(letor_output_data, 0.8,
                                                                                            lambdaFactor)


#  kMeansSpread function
#  Input : inData - Dataset
#          no_of_cluster - number of the data cluster
#  return: the centres and spreads of the clusters

def kMeansSpread(inData, no_of_clusters):
    # K-means for creating cluster.
    kmeans = KMeans(n_clusters=no_of_clusters)
    kmeans.fit(inData)
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_
    size = centroids.shape
    spreads = np.ndarray(shape=(no_of_clusters, size[1], size[1]))
    for i in range(0, no_of_clusters):
        cluster = inData[ClusterIndicesNumpy(i, labels)]
        cluster_t = cluster.transpose()
        spreads[i] = np.cov(cluster_t)
        spreads[i] = np.linalg.pinv(spreads[i])
    return spreads, centroids


# end of Kmeans Spread function

# Error function to calculate the Erms
def errorFunc(outData, weights_closed_form, designMatrix, lambdaEw):
    designMul = np.matmul(designMatrix, weights_closed_form)
    predictedVal = np.transpose(np.mat(designMul))
    error_Data = (0.5) * np.sum(np.square((outData - predictedVal)), axis=0)
    error_weight = (0.5) * lambdaEw * (np.matmul(np.transpose(weights_closed_form), weights_closed_form))
    return (np.power((2 * (error_Data + error_weight)) / (outData.shape[0]), 0.5))


# Closed form solution for Letor start
no_of_clusters = 0
validateError = 100
validateErrorMin = 100
startLetorCF = 2
endLetorCF = 25
clusterLetorCF = [0 for k in range(startLetorCF, endLetorCF)]
errorLetorCF = [0 for k in range(startLetorCF, endLetorCF)]
tErrorLetorCF = [0 for k in range(startLetorCF, endLetorCF)]
ErmsLetorCF = 0
index = 0
print("##############################")
print("      LETOR Closed Form       ")
print("##############################")

# Loop for different number of cluster within a range
# comparing the error value for the validation set for
# different clusters and selecting the optimal value for
# test data.
for no_of_clusters in range(startLetorCF, endLetorCF):
    clusterLetorCF[index] = no_of_clusters
    spreads, centroids = kMeansSpread(letor_input_data_train, no_of_clusters)

    centers = centroids[:, np.newaxis, :]
    trainData = letor_input_data_train[np.newaxis, :, :]
    designMatrixTrain = compute_design_matrix(trainData, centers, spreads)

    closed_form = closed_form_sol(lambdaFactor, designMatrixTrain, letor_output_data_train)
    print("Training using 'k' = ", no_of_clusters, " clusters")

    # Training Data Error  calculation
    tErrorLetorCF[index] = errorFunc(letor_output_data_train, closed_form, designMatrixTrain, lambdaFactor).item(0)

    validateData = letor_input_data_validate[np.newaxis, :, :]
    designMatrixValidate = compute_design_matrix(validateData, centers, spreads)

    print("Validation error for 'k' = ", no_of_clusters, " clusters")
    validateError = errorFunc(letor_output_data_validate, closed_form, designMatrixValidate, lambdaFactor)
    errorLetorCF[index] = validateError.item(0)
    index = index + 1
    print(validateError.item(0))
    if (validateErrorMin > validateError):
        validateErrorMin = validateError
        optimalClusterLetorCF = no_of_clusters
    print("--------------------------")
print("Minimum validate error = ", validateErrorMin)
print("Hence, optimal number of clusters = ", optimalClusterLetorCF)

# test data Calculation

spreadsOptimalCF, centroidsOptimalCF = kMeansSpread(letor_input_data_train, optimalClusterLetorCF)
centersOptimalCF = centroidsOptimalCF[:, np.newaxis, :]

designMatrixTrain = compute_design_matrix(trainData, centersOptimalCF, spreadsOptimalCF)
closed_form_optimal = closed_form_sol(lambdaFactor, designMatrixTrain, letor_output_data_train)
testData = letor_input_data_test[np.newaxis, :, :]
designMatrixTest = compute_design_matrix(testData, centersOptimalCF, spreadsOptimalCF)

print("Test data error for optimal number of clusters = ", optimalClusterLetorCF)
ErmsLetorCF = errorFunc(letor_output_data_test, closed_form_optimal, designMatrixTest, lambdaFactor).item(0)
print(ErmsLetorCF)

###################### LETOR SGD ########################################

no_of_clusters = 0
validateError = 100
validateErrorMin = 100
startLetorSGD = 2
endLetorSGD = 25
clusterLetorSGD = [0 for k in range(startLetorSGD, endLetorSGD)]
errorLetorSGD = [0 for k in range(startLetorSGD, endLetorSGD)]
tErrorLetorSGD = [0 for k in range(startLetorCF, endLetorCF)]
ErmsLetorSGD = 0
index = 0
print("#################################")
print("          LETOR SGD              ")
print("#################################")

# Loop for different number of cluster within a range
# comparing the error value for the validation set for
# different clusters and selecting the optimal value for
# test data.

for no_of_clusters in range(startLetorSGD, endLetorSGD):
    clusterLetorSGD[index] = no_of_clusters
    spreadsSGD, centroidsSGD = kMeansSpread(letor_input_data_train, no_of_clusters)
    centersSGD = centroidsSGD[:, np.newaxis, :]

    trainData = letor_input_data_train[np.newaxis, :, :]

    designMatrixTrain = compute_design_matrix(trainData, centersSGD, spreadsSGD)

    designMatrixValidate = compute_design_matrix(letor_input_data_validate, centersSGD, spreadsSGD)

    N, _ = designMatrixTrain.shape

    # Calculation for the SGD Weights

    SGD_weights = SGD_sol(learning_rate=learningRate, minibatch_size=N, num_epochs=10000, L2_lambda=lambdaFactor,
                          design_matrix=designMatrixTrain,
                          output_data=letor_output_data_train,
                          no_of_clusters=no_of_clusters, validData=letor_output_data_validate,
                          designMatrixValidate=designMatrixValidate)

    print("Training using 'k' = ", no_of_clusters, " clusters")

    # Training Data running
    tErrorLetorSGD[index] = errorFunc(letor_output_data_train, SGD_weights, designMatrixTrain, lambdaFactor).item(0)

    # Validate data running
    validateData = letor_input_data_validate[np.newaxis, :, :]
    designMatrixValidate = compute_design_matrix(validateData, centersSGD, spreadsSGD)

    print("Validation error for 'k' = ", no_of_clusters, " clusters")
    validateError = errorFunc(letor_output_data_validate, SGD_weights, designMatrixValidate, lambdaFactor)
    errorLetorSGD[index] = validateError.item(0)
    index = index + 1
    print(validateError.item(0))
    if (validateErrorMin > validateError):
        validateErrorMin = validateError
        optimalClusterLetorSGD = no_of_clusters
    print("--------------------------")
print("Minimum validate error = ", validateErrorMin.item(0))
print("Hence, optimal number of clusters = ", optimalClusterLetorSGD)

# test data calculation

spreadsOptimalSGD, centroidsOptimalSGD = kMeansSpread(letor_input_data_train, optimalClusterLetorSGD)
centersOptimalSGD = centroidsOptimalSGD[:, np.newaxis, :]
trainData = letor_input_data_train[np.newaxis, :, :]
designMatrixTrain = compute_design_matrix(trainData, centersOptimalSGD, spreadsOptimalSGD)
designMatrixValidate = compute_design_matrix(letor_input_data_validate, centersOptimalSGD, spreadsOptimalSGD)

SGD_weights_optimal = SGD_sol(learning_rate=learningRate, minibatch_size=N, num_epochs=10000, L2_lambda=lambdaFactor,
                              design_matrix=designMatrixTrain, output_data=letor_output_data_train,
                              no_of_clusters=optimalClusterLetorSGD, validData=letor_output_data_validate,
                              designMatrixValidate=designMatrixValidate)

testData = letor_input_data_test[np.newaxis, :, :]
designMatrixTest = compute_design_matrix(testData, centersOptimalSGD, spreadsOptimalSGD)

print("Test data error for optimal number of clusters = ", optimalClusterLetorSGD)
ErmsLetorSGD = errorFunc(letor_output_data_test, SGD_weights_optimal, designMatrixTest, lambdaFactor).item(0)
print(ErmsLetorSGD)

###################### SYNTHETIC DATA LETOR ##################################

print("#################################")
print("     SYNTHETIC Closed Form       ")
print("#################################")
# Reading the Synthetic data from the database

syn_inputData = pd.read_csv("datafiles/input.csv", header=None)
syn_outputData = pd.read_csv("datafiles/output.csv", header=None)

syn_inputData = np.array(syn_inputData)
syn_inputData_train, syn_inputData_validate, syn_inputData_test = partitionData(syn_inputData, 0.8, lambdaFactor)

syn_outputData = np.array(syn_outputData)
syn_outputData_train, syn_outputData_validate, syn_outputData_test = partitionData(syn_outputData, 0.8, lambdaFactor)

no_of_clusters = 0
validateError = 100
validateErrorMin = 100

startSynCF = 2
endSynCF = 25
clusterSynCF = [0 for k in range(startSynCF, endSynCF)]
errorSynCF = [0 for k in range(startSynCF, endSynCF)]
tErrorSynCF = [0 for k in range(startSynCF, endSynCF)]
ErmsSynCF = 0
index = 0

# Loop for different number of cluster within a range
# comparing the error value for the validation set for
# different clusters and selecting the optimal value for
# test data.

for no_of_clusters in range(startSynCF, endSynCF):
    clusterSynCF[index] = no_of_clusters
    spreads, centroids = kMeansSpread(syn_inputData, no_of_clusters)

    centers = centroids[:, np.newaxis, :]
    trainData = syn_inputData_train[np.newaxis, :, :]
    designMatrixTrain = compute_design_matrix(trainData, centers, spreads)

    closed_form = closed_form_sol(lambdaFactor, designMatrixTrain, syn_outputData_train)
    print("Training using 'k' = ", no_of_clusters, " clusters")

    # Training data calculation
    tErrorSynCF[index] = errorFunc(syn_outputData_train, closed_form, designMatrixTrain, lambdaFactor).item(0)

    # Validation step calculation
    validateData = syn_inputData_validate[np.newaxis, :, :]
    designMatrixValidate = compute_design_matrix(validateData, centers, spreads)

    print("Validation error for 'k' = ", no_of_clusters, " clusters")
    validateError = errorFunc(syn_outputData_validate, closed_form, designMatrixValidate, lambdaFactor)
    errorSynCF[index] = validateError.item(0)
    index = index + 1
    print(validateError.item(0))
    if (validateErrorMin > validateError):
        validateErrorMin = validateError
        optimalClusterSynCF = no_of_clusters
    print("--------------------------")
print("Minimum validate error = ", validateErrorMin.item(0))
print("Hence, optimal number of clusters = ", optimalClusterSynCF)

# test data calculation

spreadsOptimalCF, centroidsOptimalCF = kMeansSpread(syn_inputData_train, optimalClusterSynCF)
centersOptimalCF = centroidsOptimalCF[:, np.newaxis, :]

designMatrixTrain = compute_design_matrix(trainData, centersOptimalCF, spreadsOptimalCF)
closed_form_optimal = closed_form_sol(lambdaFactor, designMatrixTrain, syn_outputData_train)

testData = syn_inputData_test[np.newaxis, :, :]
designMatrixTest = compute_design_matrix(testData, centersOptimalCF, spreadsOptimalCF)

# closed_form = closed_form_sol(lambdaFactor, designMatrix, letor_output_data_train)
print("Test data error for optimal number of clusters = ", optimalClusterSynCF)
ErmsSynCF = errorFunc(syn_outputData_test, closed_form_optimal, designMatrixTest, lambdaFactor).item(0)
print(ErmsSynCF)

######################### Synthetic SGD #############################

print("#################################")
print("          SYNTHETIC SGD          ")
print("#################################")
no_of_clusters = 0
validateError = 100
validateErrorMin = 100

startSynSGD = 2
endSynSGD = 25
clusterSynSGD = [0 for k in range(startSynSGD, endSynSGD)]
errorSynSGD = [0 for k in range(startSynSGD, endSynSGD)]
tErrorSynSGD = [0 for k in range(startSynSGD, endSynSGD)]
ErmsSynSGD = 0
index = 0

# Loop for different number of cluster within a range
# comparing the error value for the validation set for
# different clusters and selecting the optimal value for
# test data.

for no_of_clusters in range(startSynSGD, endSynSGD):
    clusterSynSGD[index] = no_of_clusters
    spreadsSGD, centroidsSGD = kMeansSpread(syn_inputData_train, no_of_clusters)
    centersSGD = centroidsSGD[:, np.newaxis, :]

    trainData = syn_inputData_train[np.newaxis, :, :]

    designMatrixTrain = compute_design_matrix(trainData, centersSGD, spreadsSGD)

    designMatrixValidate = compute_design_matrix(syn_inputData_validate, centersSGD, spreadsSGD)

    N, _ = designMatrixTrain.shape
    # SGD Weights calculation
    SGD_weights = SGD_sol(learning_rate=learningRate, minibatch_size=N, num_epochs=10000, L2_lambda=lambdaFactor,
                          design_matrix=designMatrixTrain,
                          output_data=syn_outputData_train,
                          no_of_clusters=no_of_clusters, validData=syn_outputData_validate,
                          designMatrixValidate=designMatrixValidate)

    print("Training using 'k' = ", no_of_clusters, " clusters")

    # Training data calculation
    tErrorSynSGD[index] = errorFunc(syn_outputData_train, SGD_weights, designMatrixTrain, lambdaFactor).item(0)

    # Validation Data Calculation
    validateData = syn_inputData_validate[np.newaxis, :, :]
    designMatrixValidate = compute_design_matrix(validateData, centersSGD, spreadsSGD)

    print("Validation error for 'k' = ", no_of_clusters, " clusters")
    validateError = errorFunc(syn_outputData_validate, SGD_weights, designMatrixValidate, lambdaFactor)
    errorSynSGD[index] = validateError.item(0)
    index = index + 1
    print(validateError.item(0))
    if (validateErrorMin > validateError):
        validateErrorMin = validateError
        optimalClusterSynSGD = no_of_clusters
    print("--------------------------")
print("Minimum validate error = ", validateErrorMin.item(0))
print("Hence, optimal number of clusters = ", optimalClusterSynSGD)

# test data Error Calculation
spreadsOptimalSGD, centroidsOptimalSGD = kMeansSpread(syn_inputData_train, optimalClusterSynSGD)
centersOptimalSGD = centroidsOptimalSGD[:, np.newaxis, :]
trainData = syn_inputData_train[np.newaxis, :, :]
designMatrixTrain = compute_design_matrix(trainData, centersOptimalSGD, spreadsOptimalSGD)
designMatrixValidate = compute_design_matrix(syn_inputData_validate, centersOptimalSGD, spreadsOptimalSGD)

SGD_weights_optimal = SGD_sol(learning_rate=learningRate, minibatch_size=N, num_epochs=10000, L2_lambda=lambdaFactor,
                              design_matrix=designMatrixTrain, output_data=syn_outputData_train,
                              no_of_clusters=optimalClusterSynSGD, validData=syn_outputData_validate,
                              designMatrixValidate=designMatrixValidate)

testData = syn_inputData_test[np.newaxis, :, :]
designMatrixTest = compute_design_matrix(testData, centersOptimalSGD, spreadsOptimalSGD)

print("Test data error for optimal number of clusters = ", optimalClusterLetorSGD)
ErmsSynSGD = errorFunc(syn_outputData_test, SGD_weights_optimal, designMatrixTest, lambdaFactor).item(0)
print(ErmsSynSGD)

print("--------------------------")
print("--------------------------")
print('Optimal Number of Clusters for LETOR Data using Closed form solution = ', optimalClusterLetorCF)
print('Erms on LETOR Test Data Set using Closed Form Solution = ', ErmsLetorCF)

print('Optimal Number of Clusters for LETOR Data using SGD = ', optimalClusterLetorSGD)
print('Erms on LETOR Test Data Set using SGD = ', ErmsLetorSGD)

print('Optimal Number of Clusters for Synthetic Data using Closed form Solution = ', optimalClusterSynCF)
print('Erms on Synthetic Test Data Set using Closed Form Solution = ', ErmsSynCF)

print('Optimal Number of Clusters for Synthetic Data using SGD = ', optimalClusterSynSGD)
print('Erms on Synthetic Test Data Set using SGD = ', ErmsSynSGD)

# Graps comparison between the Closedform Solution ans stochastic Gradient descent (SGD)
# for the LETOR dataset and Synthetic Dataset

plt.figure(1)
plt.subplot(211)
plt.plot(clusterLetorCF, errorLetorCF, 'bo', clusterLetorCF, tErrorLetorCF, 'r^', linestyle='-')
plt.axis([startLetorCF - 1, endLetorCF, 0.50, 0.650])
plt.xlabel('No. of clusters')
plt.ylabel('Error closed form solution')

plt.subplot(212)
plt.plot(clusterLetorSGD, errorLetorSGD, 'bo', clusterLetorSGD, tErrorLetorSGD, 'r^', linestyle='-')
plt.axis([startLetorSGD - 1, endLetorSGD, 0.50, 0.650])
plt.xlabel('No. of clusters')
plt.ylabel('Error SGD')
plt.savefig('Letor training vs validation error lambda')
plt.show()

plt.figure(2)
plt.subplot(211)
plt.plot(clusterSynCF, errorSynCF, 'bo', clusterSynCF, tErrorSynCF, 'r^', linestyle='-')
plt.axis([startLetorCF - 1, endLetorCF, 0.65, 0.85])
plt.xlabel('No. of clusters')
plt.ylabel('Error closed form solution')

plt.subplot(212)
plt.plot(clusterSynSGD, errorSynSGD, 'bo', clusterSynSGD, tErrorSynSGD, 'r^', linestyle='-')
plt.axis([startSynSGD - 1, endSynSGD, 0.65, 0.85])
plt.xlabel('No. of clusters')
plt.ylabel('Error SGD')
plt.savefig('Synthetic data training vs validation error lambda')
plt.show()
