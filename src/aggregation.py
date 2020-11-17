import numpy as np

class AggregationMethods:
	@staticmethod
	def Mamdani(rules, fsets, outvars):
		outvars_funcs = {}
		for outvar in outvars: 
			domain = outvar.rank	
			results = []
			inputs = {key:value for key,value in fsets.items() if not value.output}
			for rule in rules:
				if fsets[rule.consequent].var.name == outvar.name: 
					results.append(rule.Evaluate({name: fset.eval() for name, fset in inputs.items()}))
			lx = []
			ly = []
			for x in np.arange(domain[0], domain[1]+0.1, 0.1):
				lx.append(x)
				y = 0
				for i in range(len(rules)):
					y = max(y, min(fsets[rules[i].consequent].eval(x), results[i]))
				ly.append(y)
			outvars_funcs[outvar.name] = (lx, ly)
		return outvars_funcs

	@staticmethod
	def Larsen(rules, fsets, outvars):
		outvars_funcs = {}
		for outvar in outvars: 
			domain = outvar.rank
			results = []
			inputs = {key:value for key,value in fsets.items() if not value.output}
			for rule in rules:
				if fsets[rule.consequent].var.name == outvar.name:
					results.append(rule.Evaluate({name: fset.eval() for name, fset in inputs.items() }))

			lx = []
			ly = []
			for i in range(len(rules)):
				for x in np.arange(domain[0], domain[1]+0.1, 0.1):
					y = fsets[rules[i].consequent].eval(x)
					z = y*results[i]
					if not x in lx:
						lx.append(x)
						ly.append(z)
					else:
						i_x = lx.index(x)
						ly[i_x] = ly[i_x] if ly[i_x] > z else z
			outvars_funcs[outvar.name] = (lx, ly)
		return outvars_funcs
