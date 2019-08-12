import numpy as np
from Analysis_Area import AreaAnalysis
import time
import matplotlib.pyplot as plt


class ClusterAnalysis:

    # def get_spaced_colors(n):
    #     max_value = 16581375  # 255**3
    #     interval = int(max_value / n)
    #     colors = [hex(I)[2:].zfill(6) for I in range(0, max_value, interval)]
    #     return tuple([(int(i[:2], 16), int(i[2:4], 16), int(i[4:], 16)) for i in colors])

    def __init__(self, ctr_points=np.asarray(list(np.genfromtxt('rsd_array_GeometricCentroids.csv', delimiter=',')) +
                                             list(np.genfromtxt('Insubstantial_structures.csv', delimiter=',')))
                 ):
        assert len(ctr_points) == 2454 + 824
        self.ctr = ctr_points

    def kmeans(self, kvalue=10, draw=False,
               export=False, analyze_area=True):

        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=kvalue, init="k-means++", n_init=10,
                    max_iter=3000, tol=1e-7, random_state=None)

        self.km_labels = km.fit_predict(self.ctr)
        self.km_error = km.inertia_
        self.km_centers = km.cluster_centers_

        if draw:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(15, 15), facecolor='0.6')

            # Draw households
            plt.scatter(self.ctr[:, 0], self.ctr[:, 1], c=self.km_labels, s=5)  # c = sequence of color
            # Draw centers
            plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
                        c=[i for i in range(kvalue)], s=100)
            plt.title("K-Means Clustering of Residences in Points \n num of Clusters: 10")
            plt.show()

        if export:
            np.savetxt("Data/Export/meanshift-total-list-plus2error.csv",
                       np.asarray(list(self.km_labels[:2447]) + [-1, -1] +
                                  list(self.km_labels[2447:])),
                       delimiter=",")
        if analyze_area:
            self.km_area_result = AreaAnalysis(labels=self.km_labels,
                                               ctr_points=self.ctr,
                                               areas=np.genfromtxt('Area_3278.csv', delimiter=','))

    def meanshift(self, bandwidth=700, draw=False,
                  export=False, analyze_area=True):
        start_time = time.time()
        from sklearn.cluster import MeanShift

        ms = MeanShift(bandwidth=bandwidth)  # bandwidth is radius

        # bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)

        ms.fit(self.ctr)
        self.ms_labels = ms.labels_
        num_clusters = len(np.unique(self.ms_labels))
        print('num of clusters:', len(np.unique(self.ms_labels)))
        self.ms_centers = ms.cluster_centers_

        if draw:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(15, 15), facecolor='.6')
            colors = ['r.', 'c.', 'b.', 'k.', 'y.', 'm.', 'g.', 'b.', 'k.', 'y.', 'm.'] * 10

            for i in range(len(self.ctr)):
                plt.plot(self.ctr[i][0], self.ctr[i][1],
                         colors[self.ms_labels[i]], markersize=3)

            # paint center points
            plt.scatter(self.ms_centers[:, 0], self.ms_centers[:, 1],
                        marker='x', s=100, linewidths=0.3, zorder=10)

            plt.title("Centers are showns as crosses \n Num of Clusters: %d"
                      % len(np.unique(self.ms_labels)))
            plt.show()

        if export:
            np.savetxt("Data/Export/meanshift-total-list-plus2error.csv",
                       np.asarray(list(self.ms_labels[:2447]) + [-1, -1] +
                                  list(self.ms_labels[2447:])),
                       delimiter=",")

        if analyze_area:
            self.ms_area_result = AreaAnalysis(labels=self.ms_labels,
                                               ctr_points=self.ctr,
                                               areas=np.genfromtxt('Area_3278.csv', delimiter=','))
        print("MeanShift Run Time:" % (time.time() - start_time))

    def DBSCAN(self, eps_value=.1, min_samples=10,
               draw=False, export=False, analyze_area=True):
        start_time = time.time()

        from sklearn.cluster import DBSCAN
        from sklearn import metrics
        from sklearn.preprocessing import StandardScaler

        X = StandardScaler().fit_transform(self.ctr)

        db = DBSCAN(eps=eps_value, min_samples=min_samples).fit(X)

        # Mark core samples
        self.db_core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        self.db_core_samples_mask[db.core_sample_indices_] = True

        self.db_labels = db.labels_

        self.db_n_clusters = len(set(self.db_labels)) - (1 if -1 in self.db_labels else 0)
        n_noise_ = list(self.db_labels).count(-1)
        print('DBSCAN: Estimated number of clusters: %d' % self.db_n_clusters)
        print('DBSCAN: Estimated number of noise points: %d' % n_noise_)
        print("DBSCAN: Silhouette Coefficient: %0.3f"
              % metrics.silhouette_score(self.ctr, self.db_labels))

        if draw:
            plt.figure(figsize=(15, 15), facecolor='.6')

            # Black removed and is used for noise instead.
            unique_labels = set(self.db_labels)
            colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
            for k, col in zip(unique_labels, colors):
                if k == -1:
                    # Black used for noise.
                    col = [0, 0, 0, 1]

                class_member_mask = (self.db_labels == k)

                xy = X[class_member_mask & self.db_core_samples_mask]
                plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                         markeredgecolor=None, markersize=5)

                xy = X[class_member_mask & ~self.db_core_samples_mask]
                plt.plot(xy[:, 0], xy[:, 1], 'x', markerfacecolor=tuple(col),
                         markeredgecolor=None, markersize=1)

            plt.title(
                'DBSCAN - num of clusters: %d \npoints with cover \
                are core point, while others are phriphery points' % len(unique_labels))

            plt.show()

        if export:
            np.savetxt("Data/Export/DBSCAN-total-list-plus2error.csv",
                       np.asarray(list(self.db_labels[:2447]) + [-1, -1] +
                                  list(self.db_labels[2447:])),
                       delimiter=",")

        if analyze_area:
            self.db_area_result = AreaAnalysis(labels=self.db_labels,
                                               ctr_points=self.ctr,
                                               areas=np.genfromtxt('Area_3278.csv', delimiter=','))

        print("DBSCAN Run Time: ", str(time.time() - start_time))


if __name__ == "__main__":
    a = ClusterAnalysis()

    # a.kmeans(kvalue=6, draw=False)
    # a.meanshift(draw=True)
    a.DBSCAN(draw=True)
