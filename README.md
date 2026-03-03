# Bident: Towards the Detection of Stealthy Unknown Traffic with Time-Context Fusion
![Example Image](overview.pdf)
# Introduction
The overall workflow of Bident sequentially incorporates data preprocessing, time-context feature fusion, single-class learners, and distribution-based adaptive threshold determination (left to right). First, the time-context feature fusion module integrates temporal and contextual features into a fusion representation and then reconstructs them to refine discriminability. Then, These reconstruction representations are fed into the single-class learners module to produce temporal and contextual reconstruction losses. Finally, the distribution-based adaptive threshold module analyzes the loss distribution to determine fine-grained detection thresholds.<br>
![Example Image](AutoEncoder.pdf)
# Requirements
pip install numpy<br>
pip install pandas<br>
pip install matplotlib<br>
pip install sklearn<br>
pip install torch<br>
pip install d2l==0.17.0<br>
# How to Use
&middot; Prepare the PCAP packets and segment the PCAP traffic into 5-tuple sessions.<br>
&middot; Run dataset_gen.py to generate the required temporal features and context features.<br>
&middot; The models.py file contains six components of the model, i.e., the Temporal Encoder, the Temporal Decoder, the Contextual Encoder, the Contextual Decoder, the Fusion Encoder and the Fusion Decoder.<br>
&middot; Run the train_test.py file to train the model and output the test results.<br>
# References
&middot; Realtime Robust Malicious Traffic Detection via Frequency Domain Analysis, Chuanpu Fu, Qi Li, Meng Shen, Ke Xu - CCS 2021.<br>
&middot; Detecting Unknown Encrypted Malicious Traffic in Real Time via Flow Interaction Graph Analysis, Chuanpu Fu, Qi Li, Ke Xu - NDSS 2023.<br>
&middot; Trident: A universal framework for fine-grained and class-incremental unknown traffic detection, Zhao Z, Li Z, Song Z - Proceedings of the ACM Web Conference 2024.<br>
&middot; Towards context-aware traffic classification via time-wavelet fusion network, Zhao Z, Song Z, Xie X - Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining 2025.
