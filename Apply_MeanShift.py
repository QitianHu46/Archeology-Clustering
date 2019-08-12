import numpy as np
from sklearn.cluster import MeanShift
import matplotlib.pyplot as plt

# from sklearn.cluster import estimate_bandwidth
rsd_ctr = np.asarray(list(np.genfromtxt('rsd_array_GeometricCentroids.csv', delimiter=',')) +
                     list(np.genfromtxt('Insubstantial_structures.csv', delimiter=',')))

assert len(rsd_ctr) == 2454 + 824

# ms = ms(bandwidth=bandwidth, bin_seeding=True)
ms = MeanShift(bandwidth=700)  # bandwidth is radius

# bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)
ms.fit(rsd_ctr)
full_label = ms.labels_
num_clusters = len(np.unique(full_label))
# np.unique() -> how many unique elements there are
print('num of clusters:', len(np.unique(full_label)))
print('center location: ', ms.cluster_centers_)


# # Generate visually distance colors
# def get_spaced_colors(n):
#     max_value = 16581375  # 255**3
#     interval = int(max_value / n)
#     colors = [hex(I)[2:].zfill(6) for I in range(0, max_value, interval)]
#     return tuple([(int(i[:2], 16), int(i[2:4], 16), int(i[4:], 16)) for i in colors])
#
#
# # Reorganize the RGB data to hex
# colors = list(get_spaced_colors(num_clusters + 1))

plt.figure(figsize=(15, 15), facecolor='.6')
colors = ['r.', 'c.', 'b.', 'k.', 'y.', 'm.', 'g.', 'b.', 'k.', 'y.', 'm.'] * 2

for i in range(len(rsd_ctr)):
    plt.plot(rsd_ctr[i][0], rsd_ctr[i][1], colors[full_label[i]], markersize=3)

# paint center points
plt.scatter(ms.cluster_centers_[:, 0], ms.cluster_centers_[:, 1],
            marker='x', s=100, linewidths=0.3, zorder=10)

plt.title("Centers are showns as crosses \n Num of Clusters: %d"
          % len(np.unique(full_label)))
plt.show()

# ====================================================================
# Area Analysis
from Analysis_Area import AreaAnalysis

aa_mean_shift = AreaAnalysis(labels=full_label,
                             ctr_points=rsd_ctr,
                             areas=np.genfromtxt('Area_3278.csv', delimiter=','))

results = aa_mean_shift.Results
