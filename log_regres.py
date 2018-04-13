from numpy import *

#load data set
def load_data_set():
	data_mat = []
	label_mat = []
	fr = open('testSet.txt')
	for line in fr.readlines():
		line_arr = line.strip().split()
		data_mat.append([1.0,float(line_arr[0]),float(line_arr[1])])
		label_mat.append(int(line_arr[2]))
	return data_mat,label_mat
 
#sigmoid(inX)
def sigmoid(inX):
	return 1.0/(1+exp(-inX))

#calculate mat
def grad_ascent(data_mat_in,class_labels):
	#change 'list' into 'numpy.matrixlib.defmatrix.matrix'
	data_matrix = mat(data_mat_in)
	label_mat = mat(class_labels).transpose()
	m,n = shape(data_matrix)
	alpha = 0.001
	max_cycles = 500
	weights = ones((n,1))
	for k in range(max_cycles):
		h = sigmoid(data_matrix*weights)
		error = (label_mat - h)
		weights = weights + alpha * data_matrix.transpose()*error
	return weights

#plot data_set and logistic function
def plot_best_fit(weights):
	import matplotlib.pyplot as plt
	#weights = wei.getA()
	data_mat,label_mat = load_data_set()
	data_arr = array(data_mat)
	n = shape(data_arr)[0]
	xcord1 = []; ycord1 = []
	xcord2 = []; ycord2 = []
	for i in range(n):
		if int(label_mat[i]) == 1:
			xcord1.append(data_arr[i,1])
			ycord1.append(data_arr[i,2])
		else:
			xcord2.append(data_arr[i,1])
			ycord2.append(data_arr[i,2])
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
	ax.scatter(xcord2, ycord2, s=30, c='green')
	x = arange(-3.0, 3.0, 0.1)
	y = (-float(weights[0]) - float(weights[1]) * x)/float(weights[2])
	ax.plot(x,y)
	plt.xlabel('X1');plt.ylabel('X2');
	plt.show()