# Intro

Site: Teotihuacan

* 'Residences.shp' is the residential spatial data for Teotihuacan
* 4 files begins with 'Apply' refers 4 different algorithms. 'Apply_Hierarchy.ipynb' is still under construction
* rsd_array++.csv is scattered endpoints of buildings. 
* rsd_array_hierarchical.csv is endpoints grouped by buildings (shapes). Only used by 'Apply_Hierarchy.ipynb'

# Outline of algorithm I have explored

1. K Means
   1. The most basic clustering algorithm. Input: no. of groups (K value), output the global/local optimum.
   2. Pro
      1. works best for data points that are separated and roughly circular
   3. Con
      1. only work for clusters that are approximately circular
2. DBSCAN
   1. An algorithm based on connectivity. Can discern continuous points witharbitrary shapes.
   2. points with profile accentuated are core points while others are phriphery points.
   3. Pro
      1. Works for any shape of cluster
      2. robust against noise
      3. no need to specify number of groups
      4. similar to the human eye functionality
   4. Con
      1. need to specify the epislon value, the radius in which two points are considered connected
3. MeanShift
   1. First set a radius (bandwidth), for each data point, compute the mean point of all the points within bandwidth, and repeat the process for the mean point. Normally, the final mean point will converge to the same place, and those whose final mean points are the same will be categorized in the same cluster.
   2. Pro
      1. no K value
      2. it's better than KMeans, because there's some degree of protection against noise
   3. Con
      1. need bandwidth or radius
4. Kernel Density Estimation (KDE)
   1. there's a density at any point on the plain - continuous function
   2. Kernel: around which the estimation is made; weights nearby events more heavily
      1. 有不同种类的kernel density function (离kernel中心不同距离不同weight)
      2. radius of function - bandwidth
   3. its relation with mean-shift
      1. mean-shift is the categorical version of KDE
5. Sequential Leader Clustering [Not used, too simple]
   1. Outline
      1. Input: threshold value to set a new group
   2. Pro
      1. really efficient, one-pass
      2. works with stream of data
   3. Con
      1. sequence of input data will affect result
      2. threshold value???
6. hierarchical clustering [not used, can be added if needed]
   1. it's rather a framework based on other clustering methods
   2. Not used. But I can apply this if necessary.



# Concerns and future improvement

## Questions

1. [**Important**]How to better account for the area of the building. Currently, I am just using the points of the polygon and it will be greatly affected by the buildign area. (like dividing a large building into two clusters) I think the main issue with the current DBSCAN clustering is because of this.
2. Civic buildings and their influences not concerned. I assume people would not live in civic buildings, but these civic buildings (like pyramids and temples) significantly affect people's behavior and routes. Do we take them into account?

## Further Development

1. Importing and processing the original data are surprisingly slow. (They take about 10 mintes) If I want to experiment with the ways to deal with area of building I may have to import raw data many times. I wonder if it's possible to grant me access to some **cloud computing service** like the Research Computing Center service provided by the school.
2. If we want case-specific clustering, **hitorical material** on the daily and public life of the city. I am glad to read some existing papers or have a conversation with the two archeologists.
3. There are some **more algorithms** to look at. I think the most fundamental two branches are K-Means and DBSCAN, which have been explored. However



# Meeting with Luis Feb 27

- for existing clusters
  - Area
    - built area
    - total area
    - number of structures
  - individual structures
    - frequency plot
      - frequency vs. area (for each cluster)
        - is buildings inside clusters more homogeneous
        - Gini coefficient on area 
        - measure validity of clustering
- Classification of buildings
  - Murals? 
  - Status 
    - e.g. insubstantial structures 
    - once have clusters: 
  - measure segregation 
- compare center of cluster with civic buildings
- Tikal
  - do clustering
  - compare with existing boundaries
- packages
  - shapely
  - fiona
  - Descartes.path



# Meeting with Luis Apr 16

Teo

- unit
- (+) linear fit loglog, area vs. area
- Insubstantial or building type?
- heat map of the density 

