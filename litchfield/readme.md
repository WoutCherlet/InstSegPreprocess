# Litchfield

## Description

1 hectare tropical plot, dominated by eucalyptus, some non-eucalypt trees and bush+grass understory.
Data collection done in August (and september post fire but august used currently)

TODO: test/train/val split?
TODO: tree density and total number in test/val/train

## Data source

Tree instances and understory tiles downloaded from UGENT shares.

Scanning details: see thesis Eliza
RIEGL VZ2000i, 20x20 regular grid, 600 khz
Filtering of points with reflectance > -15 khz and deviation < 50.

## Preprocessing

Trees were segmented manually (>5 meters) using CloudCompare.
No additional preprocessing for now.


