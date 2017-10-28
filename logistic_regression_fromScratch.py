import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def logistic_regression(x_training, y_training, x_testing, y_testing):
	
	np.random.seed(9)
	w = np.random.randn(1, x_training.shape[0])
	b= np.zeros((1,1), dtype=np.float32)
	loss_list = []
#print "w and b are:", w, b

	for i in range(60000):
		a = w.dot(x_training) + b
		y_output = 1/(1+np.exp(-a))
		#print y_output
		#loss = 1/len(x_train)*np.sum((y_train-y_output)**2, axis=1, keepdims=True)
		loss = 	np.sum(-(1/float(x_training.shape[1]))*(y_training*np.log(y_output) + (1-y_training)*np.log(1-y_output)))
		loss_list.append(loss)
		da = (1/float(x_training.shape[1]))*(y_output - y_training)
		#print da, da.shape
	
		dw = da.dot(x_training.T)
		db = np.sum(da, keepdims=True)
	
		w = w - 0.001*dw
		b = b - 0.001*db
		#print b,w
		#break
	plt.plot(range(60000), loss_list)
	a = w.dot(x_testing) + b
	y_output = np.around(1/(1+np.exp(-a))).astype(int).squeeze()
	return y_output



# import some data to play with
iris = datasets.load_iris()
X = iris.data
Y = iris.target
#print Y
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=36)

logreg = linear_model.LogisticRegression()

# we create an instance of Neighbours Classifier and fit the data.
#logreg.fit(x_train, y_train)
#print(accuracy_score(y_test, logreg.predict(x_test), normalize=True))


y_train_zero = y_train.copy()
y_test_zero = y_test.copy()

y_train_zero[y_train_zero==0] = 3
y_train_zero[y_train_zero!=3] = 0
y_test_zero[y_test_zero==0] = 3
y_test_zero[y_test_zero!=3] = 0

y_train_zero[y_train_zero==3] = 1
y_test_zero[y_test_zero==3] = 1

#print y_train_zero, y_test_zero
logreg.fit(x_train, y_train_zero)
#print logreg.predict(x_test)
print(accuracy_score(y_test_zero, logreg.predict(x_test), normalize=True))


x_train = x_train.T
x_test = x_test.T
y_train_zero = y_train_zero.reshape(x_train.shape[1], 1)
y_train_zero = y_train_zero.T

#print x_train.shape, y_train_zero.shape


print(accuracy_score(y_test_zero, logistic_regression(x_train, y_train_zero,x_test, y_test_zero ), normalize=True))

y_train_one = y_train.copy()
y_test_one = y_test.copy()

y_train_one[y_train_one==1] = 3
y_train_one[y_train_one!=3] = 0
y_test_one[y_test_zero==1] = 3
y_test_one[y_test_one!=3] = 0

y_train_one[y_train_one==3] = 1
y_test_one[y_test_one==3] = 1


#print y_train_zero, y_test_zero
logreg = linear_model.LogisticRegression()
logreg.fit(x_train.T, y_train_one)
#print logreg.predict(x_test)
print(accuracy_score(y_test_one, logreg.predict(x_test.T), normalize=True))


y_train_zero = y_train_one.reshape(x_train.shape[1], 1)
y_train_zero = y_train_one.T

#print x_train.shape, y_train_zero.shape


print(accuracy_score(y_test_one, logistic_regression(x_train, y_train_one,x_test, y_test_one ), normalize=True))





y_train_two = y_train.copy()
y_test_two = y_test.copy()

y_train_two[y_train_two==2] = 3
y_train_two[y_train_two!=3] = 0
y_test_two[y_test_two==2] = 3
y_test_two[y_test_two!=3] = 0

y_train_two[y_train_two==3] = 1
y_test_two[y_test_two==3] = 1


#print y_train_zero, y_test_zero
logreg = linear_model.LogisticRegression()
logreg.fit(x_train.T, y_train_two)
#print logreg.predict(x_test)
print(accuracy_score(y_test_two, logreg.predict(x_test.T), normalize=True))


y_train_zero = y_train_two.reshape(x_train.shape[1], 1)
y_train_zero = y_train_two.T

#print x_train.shape, y_train_zero.shape


print(accuracy_score(y_test_two, logistic_regression(x_train, y_train_two,x_test, y_test_two ), normalize=True))

plt.show()
	
