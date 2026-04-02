# Dataset Analysis Report

Dataset root: `/workspace/raw_dataset`

## Class Counts

- like: 488
- dislike: 499
- peace: 493
- background: 489

## Class Statistics

### like
- count: 488
- mean_luma: 100.09
- std_luma: 47.35
- blur_laplacian_mean: 867.40
- blur_laplacian_p10: 545.07
- entropy_mean: 6.86

### dislike
- count: 499
- mean_luma: 113.04
- std_luma: 56.43
- blur_laplacian_mean: 790.49
- blur_laplacian_p10: 663.40
- entropy_mean: 7.17

### peace
- count: 493
- mean_luma: 122.32
- std_luma: 52.41
- blur_laplacian_mean: 883.36
- blur_laplacian_p10: 701.31
- entropy_mean: 7.17

### background
- count: 489
- mean_luma: 134.84
- std_luma: 43.56
- blur_laplacian_mean: 583.76
- blur_laplacian_p10: 529.40
- entropy_mean: 6.82

## Centroid Distances (L2)

- dislike__background: 9.305
- dislike__peace: 9.534
- peace__background: 10.374
- like__peace: 11.240
- like__dislike: 13.824
- like__background: 14.607

## Duplicate Summary

- exact_duplicate_groups: 0
- exact_duplicate_images: 0
- exact_cross_class_groups: 0
- dhash_collision_groups: 277

## Recommendations

- Dataset quality looks reasonable. Proceed with training and validate with confusion matrix.
