# Object-Counting-via-Instance-Masking
Density Reconstruction via Instance Masking
In this project, we will develop a counting architecture that operates on the
principle of density completion (like Masked Auto-Encoders). Unlike standard
MAEs, that mask pixels and reconstruct, here we will mask instances and opti-
mize towards counting.
We will either count the full quantity after masking, or only the post-masking
quantity (which is an easier task).
The core hypothesis for counting the full quantity after masking is that a
model capable of accurately hallucinating the density of masked objects has
learned a robust, context-aware representation of the scene.
we will implement the following pipeline:
1. Instance-Aware Masking: Instead of random grid masking (as in stan-
dard MAE), we will use ground-truth point annotations (from the dataset
itself, pseudo-labels from bounding boxes, or aided by a Segment Anything
Model, SAM) to mask out complete object instances from the input image.
2. Density Reconstruction Objective: The model will take the masked
image as input and predict the full density map. The loss function will
explicitly penalize the error in the masked regions, forcing the network to
use context from the visible surroundings to infer the missing counts.
3. Evaluation: we will evaluate whether this pre-training or auxiliary task
improves performance on heavily occluded benchmarks compared to stan-
dard density regression.
The goals of this project are:
2
• to design a masking strategy that balances difficulty (masking enough to
require context) with feasibility,
• to analyze whether the model learns to use scene geometry (perspective)
to infer the size and density of masked objects,
• to demonstrate improved robustness to natural occlusions at test time.
Object counting models suffer significantly from domain shift. A model
trained on drone surveillance footage (e.g., VisDrone) often fails when applied
to street-level surveillance (e.g., ShanghaiTech) due to changes in perspective,
scale, and lighting.
Standard domain adaptation techniques typically require large unlabeled
target datasets (Unsupervised DA) or suffer from catastrophic overfitting when
fine-tuning on very few labeled examples. Fine-tuning a deep network on a
single labeled image usually leads to memorization: the model learns to predict
the specific global count of that image without adapting its feature extractors
to the new domain’s appearance.
In this project, we propose to treat a single labeled target image not as a sin-
gle data point, but as a generator for a combinatorial family of supervised tasks.
By applying instance-aware masking to a single labeled image, we can generate
hundreds of unique ”sub-views”, each with a known ground truth count.
