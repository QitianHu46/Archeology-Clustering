import numpy as np


# if processing raw files
if False:
	import shapefile

	path = 'residences.shp'
	sf = shapefile.Reader(path)

# assert len(sf) == 2456 # no. of shapes


# PLAIN POINTS

if False: 
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





# POINTS GROUPED WITH SHAPE

if False:
	rsd = []

	for i in range(len(sf.shapes())-2356):
		rsd.append([]) # add a shape
		for j in range(len(sf.shapes()[i].points)):
			rsd[-1].append(list(sf.shapes()[i].points[j])) # add a point in each shape

	rsd = np.asarray(rsd)
	print(rsd.shape)







# POINTS AS THE GEOMETRIC AVERAGE OF SHAPES
# ABANDONED
if False:

	raw = np.genfromtxt('rsd_array_hierarchical.csv', delimiter=',')

	def getArea(x,y):
		return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

	for i in range(2456-2356):

		area = getArea(sf.shapes()[i][:,0], sf.shapes()[i][:,1])

		rsd += []
		rsd[-1] = [
			((1/6)*area*sum([(x[a] + x[a+1])*(x[a]*y[a+1] - y[a]*x[a+1])]),
			(1/6)*area*sum([(y[a] + y[a+1])*(x[a]*y[a+1] - y[a]*x[a+1])])) for a in range(len(sf.shapes[i]))
		]

		for j in range(len(sf.shapes()[i].points)):
			rsd.append(list(sf.shapes()[i].points[j]))
		# add a new element array of all point coords to rsd
	
	rsd = np.asarray(rsd)
	np.savetxt("rsd_array_CenterOfMass.csv", rsd, delimiter=",")


	print(rsd)











