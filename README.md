# Bident: Towards the Detection of Stealthy Unknown Traffic with Time-Context Fusion
![Example Image](overview.pdf)
# Introduction
The overall workflow of Bident sequentially incorporates data preprocessing, time-context feature fusion, single-class learners, and distribution-based adaptive threshold determination (left to right). First, the time-context feature fusion module integrates temporal and contextual features into a fusion representation and then reconstructs them to refine discriminability. Then, These reconstruction representations are fed into the single-class learners module to produce temporal and contextual reconstruction losses. Finally, the distribution-based adaptive threshold module analyzes the loss distribution to determine fine-grained detection thresholds.
![Example Image](AutoEncoder.pdf)
# Requirements
pip install numpy
pip install pandas
pip install matplotlib
pip install sklearn
pip install torch
pip install d2l==0.17.0

run dataset_gen.py first,then run train_test.py
