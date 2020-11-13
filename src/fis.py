from aggregation import AggregationMethods
from defuzzification import DefuzzificationMethods
from rule import FuzzyRule
from membership import Trapezoidal, Triangular
import matplotlib.pyplot as pl
import numpy as np


class FuzzyInputVariable:
	def __init__(self, name, value, rank):
		self.name = name
		self.value = value
		self.rank = rank

class FuzzyOutputVariable:
	def __init__(self, name, rank):
		self.name = name
		self.rank = rank

class FuzzySet:
	def __init__(self, name, var, func):
		self.name = name
		self.var = var
		self.func = func
		self.output = type(var) is FuzzyOutputVariable

	def eval(self,x = None):
		if self.output:
			return self.func(x)
		return(self.func(self.var.value))	

class FuzzyInferenceSystem:
	methods = {
		'mamdani': AggregationMethods.Mamdani,
		'larsen': AggregationMethods.Larsen,
		'centroid': DefuzzificationMethods.Centroid,
		'bisector': DefuzzificationMethods.Bisection,
		'meanmax': DefuzzificationMethods.MeanMaximum,
		'lastmax': DefuzzificationMethods.LastMaximum,
		'firstmax': DefuzzificationMethods.FirstMaximum,
	}

	def __init__(self, aggrMethod, defuzMethod):
		self.aggrMethod = self.methods[aggrMethod]
		self.defuzMethod = self.methods[defuzMethod]

	def infer(self, rules, fsets, outvar):
		_rules = [FuzzyRule(rule, implication, max, min, lambda x: 1-x) for rule, implication in rules.items()]
		lx, ly = self.aggrMethod(_rules, fsets, outvar)
		return self.defuzMethod(lx, ly), lx, ly

