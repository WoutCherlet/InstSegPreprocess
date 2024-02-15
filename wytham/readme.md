# Wytham Woods

## Description

Total scanned area: 6 ha, segmented area 1,4 ha
-> down to 135 by 88 m (1,19 ha) where each point is classified (no unlabeled crowns hanging in)

Area split into 20/20/60 test/val/train.
(TODO: total number of instances in plot, total number of trees entirely in plot)
(TODO: tree density in plot)

## Data source

Tree instances downloaded from https://zenodo.org/records/7307956.
Original full plot pointcloud from UGENT shares.

Scanning details for paper described at link above.

## Preprocessing

Trees were cleaned after segmentation, so reclassification with using nearest neighbour classification was performed.
Additionally, manual cleaning of some misclassified branches and floating unclassified points was performed.

