
import numpy as np
# from sklearn.linear_model import LogisticRegression

# Calculate h(beta)
def h_beta(x, b):
	hbeta = b[0]
	for i in xrange(1,len(b)):
		hbeta += x[i-1] * b[i]
	return hbeta

# Pass h(beta) into sigmoid function to make a prediction
def Sigmoid(hbeta):
	y_hat = float(1.0 / float((1.0 + np.exp(-1 * hbeta))))
	return y_hat

# Stochastic gradient descent (SGD) to update betas
def sgd(X, y, epochs, l_rate):
	beta = np.zeros((X.shape[1]) + 1)
	for epoch in xrange(epochs):
		sum_error = 0
		for j in xrange(X.shape[0]):
			# print b
			hbeta = h_beta(X[j], beta)
			y_hat = Sigmoid(hbeta)
			error = y[j] - y_hat
			sum_error += error ** 2
			beta[0] = beta[0] + l_rate * error * y_hat * (1 - y_hat)
			for k in xrange(X.shape[1]):
				beta[k+1] = beta[k+1] + l_rate * error * y_hat * (1 - y_hat) * X[j][k]
	return beta

# Logistic Regression
def logistic_regression(X_train, y_train, X_test, y_test, epochs, l_rate):
	predictions = list()
	betas = sgd(X_train, y_train, epochs, l_rate)
	for i in xrange(X_test.shape[0]):
		hbeta = h_beta(X_test[i], betas)
		predicted = Sigmoid(hbeta)
		predicted = np.round(predicted)
		predictions.append(predicted)
	return predictions


dataset = np.array([[2.7810836,2.550537003,0],
	[1.465489372,2.362125076,0],
	[3.396561688,4.400293529,0],
	[1.38807019,1.850220317,0],
	[3.06407232,3.005305973,0],
	[7.627531214,2.759262235,1],
	[5.332441248,2.088626775,1],
	[6.922596716,1.77106367,1],
	[8.675418651,-0.242068655,1],
	[7.673756466,3.508563011,1]], np.float32)

X_train = dataset[:,0:2]
y_train = dataset[:,-1]

l_rate = 0.3

epochs = 100

l_r = logistic_regression(X_train, y_train, X_train, y_train, epochs, l_rate)
print(l_r)

# lr = logistic_regression(X_train, y_train, X_train, y_train, epochs, l_rate)
# print(lr)

# model = LogisticRegression()
# model.fit(X,y)



