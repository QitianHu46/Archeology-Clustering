class AreaAnalysis:

    def __init__(self, labels, ctr_points, areas):
        import numpy as np
        import pandas as pd

        print('area analysis working...')

        # ctr_points are all center points of buildings/insubstantial structures
        # labels are in 1-1 correspondence with ctr_points
        # areas are in 1-1 correspondence with ctr_points
        self.ctr_points = ctr_points
        self.labels = labels
        self.areas = areas
        self.num = len(ctr_points)  # number of buildings being considered
        self.df_area = pd.DataFrame(columns=['area', 'cluster'])
        self.df_area.area = self.areas
        self.df_area.cluster = self.labels

        self.num_clusters = len(set(labels))
        num_clusters = self.num_clusters

        # since DBSCAN produce label -1, need to use set(labels) rather than range(num_clusters)
        # unique_labels = list(set(labels))

        # if -1 in set(labels):
        #     num_clusters -= 1
        #     self.num_clusters -= 1


        self.Results = pd.DataFrame(columns=
                                    ['number_of_buildings', 'total_area', 'built_area', 'building_density',
                                     'max_area', 'min_2area', 'mean_area', 'Gini_coefficient'])

        # add data for cluster_number, number_of_buildings, total_area
        self.Results.number_of_buildings = [len(self.df_area[self.df_area.cluster == i]) for i in range(num_clusters)]
        self.Results.built_area = [sum(self.df_area[self.df_area.cluster == i].area) for i in range(num_clusters)]
        # self.Results.max_area = [max(self.df_area[self.df_area.cluster == i].area) for i in range(num_clusters)]
        # self.Results.min_area = [min(self.df_area[self.df_area.cluster == i].area) for i in range(num_clusters)]
        self.Results.mean_area = [self.Results.total_area[i] / self.Results.number_of_buildings[i]
                                  for i in range(num_clusters)]

        # Calculate Gini Coefficient for each cluster
        def gini(x):
            # Mean absolute difference
            mad = np.abs(np.subtract.outer(x, x)).mean()
            # Relative mean absolute difference
            rmad = mad / np.mean(x)
            # Gini coefficient
            g = 0.5 * rmad
            return g

        self.Results.Gini_coefficient = [gini(np.asarray(self.df_area[self.df_area.cluster == i].area)) for i in
                                         range(num_clusters)]

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
            convexhulls.append(ConvexHull(grouped_ctr[i]))
        self.Results.total_area = [i.volume for i in convexhulls]

        self.Results.building_density = [
            self.Results.built_area[i] / self.Results.total_area[i] for i in range(num_clusters)
        ]

        # Final Wrap-up
        self.Results.append({
            'number_of_buildings': len(ctr_points),
            'total_area': sum(self.Results.total_area),
            'built_area': sum(self.df_area.area),
            # 'max_area': max(self.df_area.area),
            # 'min_area': min(self.df_area.area),
            'mean_area': sum(self.df_area.area) / sum(self.Results.total_area),
            'Gini_coefficient': gini(np.asarray(self.df_area.area)),
            'building_density': sum(self.df_area.area) / sum(self.Results.total_area)
        }, ignore_index=True)

        self.Results.mean_area = [self.Results.total_area[i] / self.Results.number_of_buildings[i] \
                                  for i in range(num_clusters)]

    def linregres(self, fig_size=(10, 10), y_lim=(8, 16), x_lim=(10, 20)):
        from scipy import stats
        from math import log
        import matplotlib.pyplot as plt
        import numpy as np

        total_area_log = np.asarray([log(i) for i in self.Results.total_area])
        built_area_log = np.asarray([log(i) for i in self.Results.built_area])

        lin_reg = stats.linregress(x=total_area_log, y=built_area_log)
        best_fit_line = lin_reg.slope * np.asarray([i for i in range(30)]) + lin_reg.intercept

        plt.figure(facecolor='.6', figsize=fig_size)
        plt.scatter(x=total_area_log, y=built_area_log)
        plt.plot(best_fit_line)
        plt.ylim(y_lim)
        plt.xlim(x_lim)
        plt.xlabel("logged total area of each cluster")
        plt.ylabel("logged built area of each cluster")
        plt.title('Graph of built area against total area (both logged)')
        plt.show()
        print('slope of the line is: ', lin_reg.slope)
        print('y-intercept of the line is: ', lin_reg.intercept)


# if __name__=="__main__":
#     k = AreaAnalysis(labels=[1,2,3], ctr_points=[[3,4],[56,0],[3,6]], areas=[5,2,5])