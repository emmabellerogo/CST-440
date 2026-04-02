# Dataset Analysis Report

Dataset root: `/workspace/raw_dataset`

## Class Counts

- like: 386
- dislike: 362
- peace: 369
- background: 248

## Class Statistics

### like
- count: 386
- mean_luma: 121.71
- std_luma: 36.52
- blur_laplacian_mean: 598.47
- blur_laplacian_p10: 455.38
- entropy_mean: 6.58

### dislike
- count: 362
- mean_luma: 116.44
- std_luma: 41.27
- blur_laplacian_mean: 695.51
- blur_laplacian_p10: 590.90
- entropy_mean: 6.90

### peace
- count: 369
- mean_luma: 111.46
- std_luma: 42.00
- blur_laplacian_mean: 160.78
- blur_laplacian_p10: 131.05
- entropy_mean: 6.87

### background
- count: 248
- mean_luma: 122.40
- std_luma: 31.88
- blur_laplacian_mean: 195.35
- blur_laplacian_p10: 5.07
- entropy_mean: 5.91

## Centroid Distances (L2)

- like__background: 6.783
- dislike__background: 7.615
- like__dislike: 9.899
- peace__background: 10.863
- dislike__peace: 11.269
- like__peace: 11.644

## Duplicate Summary

- exact_duplicate_groups: 0
- exact_duplicate_images: 0
- exact_cross_class_groups: 0
- dhash_collision_groups: 98

## Recommendations

- background: many low-detail frames (blur p10=5.07). Remove very blurry samples.
- Background confusion is high (20.2% to gesture classes). Add harder negative/background samples.
