library(tidyverse)
library(sf)
library(lwgeom)
library(here)

# Use st_crs() to check projection method
# Use st_transform() to chenge to another projection method

# parcel <- st_read(here("Data", "Parcels_for_Yuna", "parcellanduse.shp"))

parcel <- st_read("/Users/jasonhu/Downloads/R_Wrangling/Data/Parcels_for_Yuna/parcellanduse.shp") %>% 
  st_transform(3435) %>% 
  st_make_valid() 
# This customs data seems to have invalid projection, and need to use here::st_make_valid to make it work

neigh <- st_read("/Users/jasonhu/Downloads/R_Wrangling/Data/Boundaries - Neighborhoods/geo_export_1a667356-4a30-424e-89f3-4ff570ed08f1.shp") %>% 
  select(pri_neigh) %>% 
  st_transform(3435)

parcel_combined <- st_join(parcel, neigh, join = st_within)

zoning <- st_read("/Users/jasonhu/Downloads/R_Wrangling/Data/Boundaries - Zoning Districts (current)/geo_export_269b532d-7b4d-4709-a1bb-df254db5f55e.shp") %>% 
  select(zone_class, zoning_id, zone_type) %>% 
  st_transform(3435) %>% 
  st_make_valid()


parcel_combined <- st_join(parcel_combined, zoning, join = st_within)
parcel_combined <- st_make_valid(parcel_combined) # not sure if necessary

st_write(parcel_combined, "parcel_combined.geoJSON")
