library(tidyverse)
library(here)
library(sf)

rsd_cluster_list <- read_csv(here("Export", "kmeans-resident-list.csv"), col_names = "cluster")


a <- st_read("/Users/qitianhu/Documents/GitHub/Teo-Clustering/Data/insubs_Teo/Small_Features.shp")


rsd_data <- st_read(here("Data", "Residences.shp"))

rsd_data2 <- left_join(rsd_data, rsd_cluster_list, by = "cluster")

# I cannot just directly cut them out
# rsd_data <- rsd_data[-c(2448:2449)]

rsd_data %>%
  # remove rows 2448 and 2449 (in Python 2447, 2448)
  filter(Apt_sub_ID != "Estructura 69")
# mutate(kmeans_group = data.frame(rsd_cluster_list))

methods(class = "sf")

# world_agg1 = aggregate(pop ~ continent, FUN = sum, data = world, na.rm = TRUE)
