#test code of logistic
from numpy import *
import log_regres

def test():
	dataArr,labelMat = log_regres.load_data_set()
	# weights = log_regres.grad_ascent(dataArr,labelMat)
	# weights = log_regres.stoc_grad_ascent0(array(dataArr),labelMat)
	weights = log_regres.stoc_grad_ascent1(array(dataArr),labelMat)
	log_regres.plot_best_fit(weights)

test()