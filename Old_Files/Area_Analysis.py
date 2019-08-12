class AreaAnalysis:

    def __init__(self, labels, ctr_points, areas):
        print('setoff')

        import numpy as np
        import fiona
        from shapely.geometry import Point, Polygon, shape
        import pandas as pd


        # ctr_points are all center points of buildings/insubstantial structures
        # labels are in 1-1 correspondence with ctr_points
        # areas are in 1-1 correspondence with ctr_points
        self.ctr_points = ctr_points
        self.labels = labels
        self.areas = areas
        self.num_clusters = len(set(labels))
        num_clusters = self.num_clusters
        self.num = len(ctr_points) # number of buildings being considered
        dfArea = pd.DataFrame(columns = ['area', 'cluster'])
        dfArea.area = self.areas 
        dfArea.cluster = self.labels

        self.Results = pd.DataFrame(columns = 
        ['number_of_buildings', 'total_area', 'built_area', 'building_density',
         'max_area', 'min_area', 'mean_area', 'Gini_coefficient' ])

        # add data for cluster_number, number_of_buildings, total_area
        self.Results.number_of_buildings = [len(dfArea[dfArea.cluster == i]) for i in range(num_clusters) ]
        self.Results.built_area = [sum(dfArea[dfArea.cluster == i].area) for i in range(num_clusters) ]
        # self.Results.max_area = [max(dfArea[dfArea.cluster == i].area) for i in range(num_clusters) ]
        # self.Results.min_area = [min(dfArea[dfArea.cluster == i].area) for i in range(num_clusters) ]
        self.Results.mean_area = [ self.Results.total_area[i]/self.Results.number_of_buildings[i] \
                                    for i in range(num_clusters) ]

        # Calculate Gini Coefficient for each cluster
        def gini(x):
            # Mean absolute difference
            mad = np.abs(np.subtract.outer(x, x)).mean()
            # Relative mean absolute difference
            rmad = mad/np.mean(x)
            # Gini coefficient
            g = 0.5 * rmad
            return g

        self.Results.Gini_coefficient = [gini(np.asarray(dfArea[dfArea.cluster == i].area)) for i in range(num_clusters)]


        # Calculate total area of each cluster
        # use the area convexHull of all center in the cluster
        # http://scipy.github.io/devdocs/generated/scipy.spatial.ConvexHull.html
        grouped_ctr = [[] for i in range(num_clusters)]
        for i in range(len(ctr_points)):
            grouped_ctr[labels[i]].append(ctr_points[i])

        # each item in convexhulls is a ConvexHull object consisted of the cluster
        from scipy.spatial import ConvexHull
        convexhulls = []
        for i in range(num_clusters):
            convexhulls.append(ConvexHull(grouped_ctr[i]) )
        self.Results.total_area = [i.volume for i in convexhulls]

        self.Results.building_density = [
            self.Results.built_area[i]/self.Results.total_area[i] for i in range(num_clusters)
        ]

        # Final Wrap-up
        self.Results.append({
            'number_of_buildings': len(ctr_points), 
            'total_area': sum(self.Results.total_area), 
            'built_area': sum(dfArea.area), 
            'max_area': max(dfArea.area), 
            'min_area': min(dfArea.area),
            'mean_area': sum(dfArea.area)/sum(self.Results.total_area), 
            'Gini_coefficient': gini(np.asarray(dfArea.area)),
            'building_density': sum(dfArea.area)/sum(self.Results.total_area)
        }, ignore_index=True)

        self.Results.mean_area = [ self.Results.total_area[i]/self.Results.number_of_buildings[i] \
                            for i in range(num_clusters) ]

    def printing(self):
        print('it\'s working')

    def linRegres(self, fig_size = (10, 10), y_lim = (8, 16), x_lim = (10,20)):
        from scipy import stats
        from math import log

        total_area_log = np.asarray([log(i) for i in self.Results.total_area])
        built_area_log = np.asarray([log(i) for i in self.Results.built_area])

        lin_reg = stats.linregress(x = total_area_log, y = built_area_log)
        best_fit_line = lin_reg.slope * np.asarray([i for i in range(30)]) + lin_reg.intercept

        figure10 = plt.figure(facecolor = '.6', figsize = fig_size)
        plt.scatter(x = total_area_log, y = built_area_log)
        plt.plot(best_fit_line)
        plt.ylim(y_lim); plt.xlim(x_lim)
        plt.xlabel("logged total area of each cluter")
        plt.ylabel("logged built area of each cluter")
        plt.title('Graph of built area against total area (both logged)')
        plt.show()
        print('slopt of the line is: ',lin_reg.slope)
        print('y-intercept of the line is: ',lin_reg.intercept)



# import numpy as np
# import fiona
# from shapely.geometry import Point, Polygon, shape
# import pandas as pd
# import random

# rsd_ctr = np.asarray(list(np.genfromtxt('rsd_array_GeometricCentroids.csv', delimiter=',')) + 
#             list(np.genfromtxt('Insubstantial_structures.csv', delimiter=',')))

# # random_list = [1 if i//3 == 0 else 2 for i in range(len(rsd_ctr)) ]
# random_list_cluster = np.asarray([random.randint(0,10) for i in range(len(rsd_ctr))])
# random_list_area = np.asarray([random.randint(0,10) for i in range(len(rsd_ctr))])

# sth = AreaAnalysis2(ctr_points = rsd_ctr,
#                     labels = random_list_cluster,
#                     areas = random_list_area)



# print(sth.Results)










# # DEBUGGER
# num_clusters = len(set(random_list_area))


# dfArea = pd.DataFrame(columns = ['area', 'cluster'])
# dfArea.area = random_list_area
# dfArea.cluster = random_list_cluster

# Results = pd.DataFrame(columns = 
#     ['number_of_buildings', 'total_area', 'built_area', 'building_density',
#      'max_area', 'min_area', 'mean_area', 'Gini_coefficient' ])


# area_list = []
# for i in range(num_clusters):
#     area_list.append([])
#     for j in range(len(rsd_ctr)):
#         if dfArea.cluster[j] == i:
#             area_list[-1].append(dfArea.area[j])


# Results.max_area = [max(area_list[i]) for i in range(num_clusters)]

# # Results.max_area = [max( dfArea[dfArea.cluster == i].area ) for i in range(num_clusters) ]




