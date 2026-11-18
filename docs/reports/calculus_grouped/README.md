# Conservative grouped calculus comparison

Notebook 14.8 executed locally. Group edges use exact SHA-256, matching pre-Roboflow filename or dHash distance <=4, including transitive connections. Keep historical validation and exclude all connected training members. Similarity candidates can over-group distinct scenes; patient independence remains unverified. Within-validation repetitions remain.

| Membership | Images | Boxes | Components |
|---|---:|---:|---:|
| Training retained | 350 | 3022 | 201 |
| Validation retained | 239 | 1498 | 145 |
| Training excluded | 496 | 2876 | 108 |

Manifest SHA-256: `831e8768df7ed11d7608b008297380a2014be4a1badeb56c21a1d0f70e828a50`. Membership, edges and image/label checksums are saved alongside this report. Candidate montage was visually inspected and shows repeated scenes with different names. Test remains unopened.

14.9n and 14.9s embed the same manifest. Both verify source image and normalized calculus-label bytes, use Singh868 v1, Ultralytics 8.3.228, 640 input, batch16, seed42, 60 epochs, patience0, AdamW lr0=0.001 weight_decay=0.0005 and identical augmentation. Five-hour epoch-boundary guard, periodic checkpoints every10 epochs, one GPU worker and requested NvidiaTeslaT4 per notebook. AdamW is a deliberate matched protocol choice and differs from historical automatic optimizer selection. Historical scores also use different training membership.

Public Kaggle version1 submissions: `trihuynhviprovcl/dentalselfie-grouped-v8n-20261117` and `trihuynhviprovcl/dentalselfie-grouped-v8s-20261117`. Initial live states RUNNING and QUEUED respectively. Actual accelerator awaits logs. No new dataset upload required.

Next collect both outputs and inspect small-box misses before a conditional P2 experiment. Positive-only validation cannot measure negative-image specificity or establish phone performance. Historical 78.03% mAP50 is not a matched reference for this changed split.
