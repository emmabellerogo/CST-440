# Dataset Analysis Report

Dataset root: `/workspace/raw_dataset`

## Class Counts

- like: 362
- dislike: 352
- peace: 360
- background: 367

## Class Statistics

### like
- count: 362
- mean_luma: 135.00
- std_luma: 61.44
- blur_laplacian_mean: 1996.44
- blur_laplacian_p10: 1398.91
- entropy_mean: 7.51

### dislike
- count: 352
- mean_luma: 142.22
- std_luma: 67.06
- blur_laplacian_mean: 2809.05
- blur_laplacian_p10: 2533.74
- entropy_mean: 7.57

### peace
- count: 360
- mean_luma: 150.18
- std_luma: 57.72
- blur_laplacian_mean: 1478.26
- blur_laplacian_p10: 1241.27
- entropy_mean: 7.44

### background
- count: 367
- mean_luma: 149.59
- std_luma: 45.43
- blur_laplacian_mean: 1274.67
- blur_laplacian_p10: 443.01
- entropy_mean: 7.22

## Centroid Distances (L2)

- peace__background: 8.281
- dislike__background: 10.047
- dislike__peace: 10.732
- like__peace: 11.776
- like__background: 11.949
- like__dislike: 12.418

## Duplicate Summary

- exact_duplicate_groups: 0
- exact_duplicate_images: 0
- exact_cross_class_groups: 0
- dhash_collision_groups: 227

## Recommendations

- Dataset quality looks reasonable. Proceed with training and validate with confusion matrix.
