import shapefile
import numpy as np


path = 'residences.shp'
sf = shapefile.Reader(path)

# assert len(sf) == 2456 # no. of shapes


# PLAIN POINTS
'''
# Import datapoints to numpy array
# for now, just separate the points
rsd = []
for i in range(2456):
	for j in range(len(sf.shapes()[i].points)):
		rsd.append(list(sf.shapes()[i].points[j]))
	# add a new element array of all point coords to rsd
rsd = np.asarray(rsd)

np.savetxt("rsd_array++.csv", rsd, delimiter=",")
print(len(rsd))
'''


# POINTS GROUPED WITH SHAPE

rsd = []

for i in range(len(sf.shapes())-2356):
	rsd.append([]) # add a shape
	for j in range(len(sf.shapes()[i].points)):
		rsd[-1].append(list(sf.shapes()[i].points[j])) # add a point in each shape

rsd = np.asarray(rsd)


print(rsd[].shape)

'''
import pandas as pd # numpy cannot save 3D array, and use pandas dataframe instead
stacked = pd.Panel(rsd)
# save to disk
stacked.to_csv('rsd_array_hierarchical.csv')
'''