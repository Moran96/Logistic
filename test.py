#test code of logistic
from numpy import *
import log_regres

def test():
	dataArr,labelMat = log_regres.load_data_set()
	weights = log_regres.grad_ascent(dataArr,labelMat)
	log_regres.plot_best_fit(weights.getA())

test()