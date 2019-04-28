class AreaAnalysis:

    import numpy as np
    import fiona
    from shapely.geometry import Point, Polygon, MultiPolygon, shape
    import pandas as pd

    file_path = 'Data/Residences.shp'
    rsd_ctr = np.asarray(list(np.genfromtxt('rsd_array_GeometricCentroids.csv', delimiter=',')) + 
            list(np.genfromtxt('Insubstantial_structures.csv', delimiter=',')))

    def __init__(self, labels, ctr_points = rsd_ctr):

        import numpy as np
        import fiona
        from shapely.geometry import Point, Polygon, MultiPolygon, shape
        import pandas as pd


        assert len(ctr_points) == len(labels) and len(ctr_points[0]) == 2
        self.ctr_points = ctr_points
        self.labels = labels
        self.num_clusters = len(set(labels))

        rawMap = fiona.open('Data/Residences.shp')

        # assort all the geoPy objects in the geoMap list
        self.geoMap = []
        for i in range(len(rawMap)):
            if rawMap[i]['geometry'] == None:
                print(i, 'is None')
            else:
                try:
                    self.geoMap.append(Polygon(shape(rawMap[i]['geometry'])))
                except:
                    print(i, 'has error')

        self.dfArea = pd.DataFrame(columns = ['area', 'cluster'])
        self.dfArea.area = [m.area for m in self.geoMap] + [25 for i in range(len(ctr_points) - len(rawMap))]
        self.dfArea.cluster = self.labels
        print(self.dfArea.head())



# # Setup the result DataFrame
# Results = pd.DataFrame(columns = 
# ['number_of_buildings', 'total_area', 'built_area', 'building_density',
#  'max_area', 'min_area', 'mean_area', 'Gini_coefficient' ])


# # add data for cluster_number, number_of_buildings, total_area
# Results.number_of_buildings = [len(dfArea[dfArea.cluster == i]) for i in range(num_clusters) ]
# Results.built_area = [ sum(dfArea[dfArea.cluster == i].area) for i in range(num_clusters) ]
# Results.max_area = [ max(dfArea[dfArea.cluster == i].area) for i in range(num_clusters) ]
# Results.min_area = [ min(dfArea[dfArea.cluster == i].area) for i in range(num_clusters) ]
# Results.mean_area = [ Results.total_area[i]/Results.number_of_buildings[i] for i in range(num_clusters) ]

# # Calculate Gini Coefficient for each cluster
# def gini(x):
#     # Mean absolute difference
#     mad = np.abs(np.subtract.outer(x, x)).mean()
#     # Relative mean absolute difference
#     rmad = mad/np.mean(x)
#     # Gini coefficient
#     g = 0.5 * rmad
#     return g

# Results.Gini_coefficient = [ gini(np.asarray(dfArea[dfArea.cluster == i].area)) for i in range(num_clusters) ]


# # Calculate total area of each cluster region
# # use the area convexHull of all center in the cluster
# # http://scipy.github.io/devdocs/generated/scipy.spatial.ConvexHull.html
# grouped_ctr = [[] for i in range(num_clusters)]
# for i in range(len(rsd_ctr)):
#     grouped_ctr[labels[i]].append(rsd_ctr[i])





# # each item in convexhulls is a ConvexHull object consisted of the cluster
# from scipy.spatial import ConvexHull

# convexhulls = []
# for i in range(num_clusters):
#     convexhulls.append(ConvexHull(grouped_ctr[i]) )

# Results.total_area = [ i.volume for i in convexhulls]

# Results.built_area = [
#     sum(dfArea[dfArea.cluster == i].area) for i in range(num_clusters)
# ]


# Results.building_density = [
#     Results.built_area[i]/Results.total_area[i] for i in range(num_clusters)
# ]


# Results.append({
#     'number_of_buildings': len(rsd_ctr), 'total_area': sum(Results.total_area), 
#     'built_area': sum(dfArea.area), 'max_area': max(dfArea.area), 'min_area': min(dfArea.area),
#     'mean_area': sum(dfArea.area)/sum(Results.total_area), 'Gini_coefficient': gini(np.asarray(dfArea.area)),
#     'building_density': sum(dfArea.area)/sum(Results.total_area)
# }, ignore_index=True)
