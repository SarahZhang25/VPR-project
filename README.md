# Visual Place Recognition Project: Evaluation of AnyLoc

This repository contains code to evaluate the performance of AnyLoc on the Garden's Point dataset, as well as for running SIFT/bag of visual words (BoVW) on the same dataset as a baseline for comparison against AnyLoc.

Architecture comparison: AnyLoc uses a specific layer/facet of the DINOv2 foundation model for per-pixel features and applies unsupervised local feature aggregation methods (VLAD) for place-level global descriptors. Thus, we are comparing the DINOv2 features against SIFT and VLAD against BoVW.


### Dataset
Source: https://zenodo.org/records/4590133#.ZAdKktJBxH5. 

The dataset images are also contained in the `gardens.zip` file on the repo's home directory, which should be unzipped into a folder named `./gardens`. The dataset itself consists of 3 traverses through a college campus, and contains a range of environments including indoor, urban, and outdoor/natural scenes, which makes it a challenging dataset. Two traverses are daytime traverses on different sides of a walkway, and one is a nighttime traverse. The night images have been extremely contrast enhanced and converted to grayscale during processing. It contains 600 total images at 960 x 540 resolution, and uses a 2 frame tolerance for ground truth evaluation (`./gardens/gardens_gt.npy`). We use the `day_right` subset as the database/training set and the `day_left` and `night_right` sets as the query/test set. These test sets allow us to analyze robustness to changes in viewpoint and lighting, respectively.

## AnyLoc analysis
All work with AnyLoc is contained in the `anyloc_gardens_eval.ipynb` notebook, which is set up to run in Google Colab with a GPU runtime. This notebook is built off the demo notebook provided by AnyLoc authors found [here](https://github.com/AnyLoc/AnyLoc/blob/main/demo/anyloc_vlad_generate_colab.ipynb).

The vocabularies (e.g. cluster centers for VLAD aggregation) provided by AnyLoc authors are found in `vocabulary/`. We primarily compare `vocabulary/dinov2_vitg14/l31_value_c32/urban/c_centers.pt` and `vocabulary/dinov2_vitg14/l31_value_c32/indoor/c_centers.pt` against a custom vocabulary derived from the `day_right` image set. This allows us to evaluate how the non-dataset-specific vocabularies perform against a dataset-specific one. Top-k retrieval for an image query is performend via [faiss](https://github.com/facebookresearch/faiss).

## SIFT/BoW analysis
All work with SIFT/BoW is contained in the `sift_bow_gardens_eval.ipynb` notebook. We use OpenCV to extract SIFT features, and construct BoW descriptors based of k=32 and k=100 clusters (whose centers are learned from the `day_right` image set). Top-k retrieval for an image query is performed via KNN. 


## Loading in pre-computed descriptors
Both notebooks contain code for generating the descriptors for all images from scratch based on tunable parameters and saving them all into a `cache/` folder. To get off the ground faster, a fully populated cache can be found and downloaded at the following link: [https://drive.google.com/file/d/1DTrGavbyYrPTPCTUTY5bD0MOVHQ914Qv/view?usp=sharing](https://drive.google.com/file/d/1DTrGavbyYrPTPCTUTY5bD0MOVHQ914Qv/view?usp=sharing). 

This folder should be unzipped, saved into a folder named `cache`, and placed in the home directory.

# Results
Tables for experimental results are found in `results_tables.xlsx`. Key takeaways of AnyLoc evaluations:
* We see perfect recall@5 and almost perfect recall@1 on both test sets, showing robust performance to environmental changes. 
* The nighttime test set had slightly better resulsts than the viewpoint shift test set, which suggests the model is more robust against lighting than viewpoint changes.  
* The generalized domain vocabularies are competitive (only a couple percentage points worse) against a dataset-specific one, showing these provided vocabularies are robust for VPR (as claimed by the paper). 

Key takeaways of comparison against SIFT/BoW:
* SIFT/BoW struggled significantly, achieving <0.2 Recall@5 on the viewpoint shift test set and ~.85 Recall@5 on the nighttime test set. 
* AnyLoc significantly outperforms SIFT/BoVW in terms of accuracy, but from a runtime and resource usage perspective, SIFT/BoW remains a winner: SIFT/BoW (on CPU) runs multiple orders of magnitude faster than AnyLoc (on a Google Colab with an NVidia T4 GPU)