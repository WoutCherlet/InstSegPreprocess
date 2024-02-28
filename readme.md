# Instance segmentation benchmark data preprocessing

## Description

All preprocessing code for benchmark dataset for instance segmentation.
Some of the common preprocessing steps for this data include:
- visualization of understory and trees
- overlapping of trees back onto original tiles
- reclassifying tree points using nearest neighbour/distance based classification
- test/val/train/split
- preparing manual segmentation

4 sites are used:
- Litchfield, AUS
- Wytham, UK
- Offenthal, GER
- Robson Creek, AUS

Each site is fully segmented into understory/tree points, with a unique instance label assigned to each tree.

Additional info on data source and preprocessing can be found in the seperate folders.


## File format of output

All files are in the ply format.
Instances labels are denoted in a scalar field called 'instance'. Tree points have a unique label starting at 1, understory points have an instance label of -1.
Semantic labels are stored in a scalar field called 'semantic'. Understory points are labeled 0 and tree points are labeled 1.

The full test/validation/train areas are provided as single files.
Additionally, tiling can be performed for methods operating on smaller areas. Overlap between tiles is optional.

All trees that are present in the test area are stored in a seperate folder.
These are further divided into eval and non-eval trees using a threshold on the fraction of points lying within the plot. Currently, 0.9 is used for this threshold.