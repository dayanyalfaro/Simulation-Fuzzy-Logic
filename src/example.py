from fis import *
from membership import *
import matplotlib.pyplot as pl
import numpy as np


temperature = FuzzyInputVariable('Temperature',30,(10,40))
humidity = FuzzyInputVariable('Humidity', 60, (20, 100))

speed = FuzzyOutputVariable('Speed',(0,100))

rules = {
# If Temperature is Cold and Humidity is Wet Then Speed is Slow
'ColdTemperature & WetHumidity': 'SlowSpeed',
# If Temperature is Medium and Humidity is Wet Then Speed is Slow
'MediumTemperature & WetHumidity': 'SlowSpeed',
# If Temperature is Cold and Humidity is Normal Then Speed is Slow
'ColdTemperature & NormalHumidity': 'SlowSpeed',
# If Temperature is Hot and Humidity is Wet Then Speed is Moderate
'HotTemperature & WetHumidity': 'ModerateSpeed',
# If Temperature is Medium and Humidity is Normal Then Speed is Moderate
'MediumTemperature & NormalHumidity': 'ModerateSpeed',
# If Temperature is Cold and Humidity is Dry Then Speed is Moderate
'ColdTemperature & DryHumidity': 'ModerateSpeed',
# If Temperature is Hot and Humidity is Normal Then Speed is Fast
'HotTemperature & NormalHumidity': 'FastSpeed',
# If Temperature is Hot and Humidity is Dry Then Speed is Fast
'HotTemperature & DryHumidity': 'FastSpeed',
# If Temperature is Medium and Humidity is Dry Then Speed is Fast
'MediumTemperature & DryHumidity': 'FastSpeed',
}


fuzzy_sets = {
	'ColdTemperature': FuzzySet('Cold',temperature,Triangular(10, 25, 10)),
	'MediumTemperature': FuzzySet('Medium',temperature,Triangular(15, 35, 25)),
	'HotTemperature': FuzzySet('Hot',temperature,Triangular(25, 40, 40)),
	'WetHumidity': FuzzySet('Wet',humidity,Triangular(20, 60, 20)),
	'NormalHumidity': FuzzySet('Normal',humidity,Trapezoidal(30, 45, 75,90)),
	'DryHumidity': FuzzySet('Dry',humidity,Triangular(60, 100, 100)),
	'SlowSpeed': FuzzySet('Slow',speed,Triangular(0, 50, 0)),
	'ModerateSpeed': FuzzySet('Moderate',speed,Triangular(10, 90, 50)),
	'FastSpeed': FuzzySet('Fast',speed,Triangular(50, 100, 100))
}


s = FuzzyInferenceSystem('mamdani', 'centroid')
y, lx, ly = s.infer(rules,  fuzzy_sets, speed)
print('mamdani', 'centroid',y)
pl.plot(lx, ly)
pl.show()

s = FuzzyInferenceSystem('mamdani', 'bisector')
y, lx, ly = s.infer(rules, fuzzy_sets, speed)
print('mamdani', 'bisector', y)

s = FuzzyInferenceSystem('mamdani', 'meanmax')
y, lx, ly = s.infer(rules,fuzzy_sets, speed)
print('mamdani', 'meanmax', y)

s = FuzzyInferenceSystem('mamdani', 'lastmax')
y, lx, ly = s.infer(rules, fuzzy_sets, speed)
print('mamdani', 'lastmax', y)

s = FuzzyInferenceSystem('mamdani', 'firstmax')
y, lx, ly = s.infer(rules, fuzzy_sets, speed)
print('mamdani', 'firstmax', y)

s = FuzzyInferenceSystem('larsen', 'centroid')
y, lx, ly = s.infer(rules, fuzzy_sets, speed)
print('larsen', 'centroid',y)

s = FuzzyInferenceSystem('larsen', 'bisector')
y, lx, ly = s.infer(rules,  fuzzy_sets, speed)
print('larsen', 'bisector', y)

s = FuzzyInferenceSystem('larsen', 'meanmax')
y, lx, ly = s.infer(rules,  fuzzy_sets, speed)
print('larsen', 'meanmax', y)

s = FuzzyInferenceSystem('larsen', 'lastmax')
y, lx, ly = s.infer(rules,fuzzy_sets, speed)
print('larsen', 'lastmax', y)

s = FuzzyInferenceSystem('larsen', 'firstmax')
y, lx, ly = s.infer(rules, fuzzy_sets, speed)
print('larsen', 'firstmax', y)
pl.plot(lx, ly)
pl.show()