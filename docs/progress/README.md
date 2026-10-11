# Progress log

A record of what worked, what didn't, and the numbers behind each change.

## Piece detector (YOLO11s, imgsz 1024)

| version | change | mAP50-95 | city recall | notes |
|---|---|---|---|---|
| v1 | trained on Rounds 1-2 (39 images) | 0.814 | 0.783 | scored against Round 3 labels that were never reviewed |
| v1 | same model, Round 3 labels reviewed | 0.844 | 0.808 | the model was better than its first score: many "errors" were label errors |
| v2 | Rounds 1-2 relabeled, spare-piece piles masked with an `ignore` class | 0.848 | **0.932** | the gain came from data quality, not model changes |

- `detector_v1_training_curves.png`: v1 converged by about epoch 60, with no overfitting.
- `detector_v1_confusion_matrix_conf0001.png`: the matrix Ultralytics saves during training uses conf 0.001, so it is full of junk boxes. Recomputed at conf 0.25, the real problem was cities predicted as settlements.
- Main remaining detector weakness: city vs settlement when the tower step isn't visible (top-down or tower facing the camera).

## Rectification

- 6 corner clicks per photo (middle of each corner tile's outer edge), homography to a 1000 px ideal board.
- Corner fit error: mean 2-7 px, worst 10.4 px on a board with 100 px tiles. Tolerance for snapping is about 50 px.
- `corner_clicks_round3.jpg`, `rectify_grid_overlay.jpg`, `board_vertex_edge_ids.jpg`.

## First unseen photo (test01)

Different room, daylight, tablecloth, 3 players, steep angle. Nothing like it was in training.

- Detector + snapping: 39 roads, 10 settlements, 4 cities, 1 robber on the board, all correct by hand count.
- Spare pieces and one false positive on the tablecloth were dropped automatically by the snap distance check.
- `detector_test01_unseen_room.jpg`.

## Color classification

| version | method | accuracy | where it failed |
|---|---|---|---|
| k-means | 3 color clusters per crop in Lab, nearest of 4 reference colors learned from 198 hand labels | 93% on Round 3, 91% on all labeled crops | shadowed orange read as red; brick tile behind a blue or white piece read as orange; cream white under warm light |
| learned | YOLO11n-cls on 2,071 reviewed crops (k-means pre-labels, 173 corrected by hand) | **99.8% on Round 3** (100% at best epoch) | none seen yet |

Tried and rejected for k-means: down-weighting lightness (inconsistent), using only the center cluster (no gain), K = 2, 4, 5 (K = 3 was best).

- `color_kmeans_round3_misses.jpg`: the 3 Round 3 misses (2 shadowed orange settlements read as red, 1 blue city read as white).
- `color_review_page_white.jpg`: one review page of crops k-means called white. Several are clearly orange, red, or blue.
- `color_before_after_IMG_8601.jpg`: setup phase, every player should have 2 settlements and 2 roads. k-means gave orange 4/4 and white 2/0. The learned classifier gets 2/2 for every player.
- `color_before_after_test01.jpg`: k-means gave red 9 roads and orange 16. The learned classifier gives red 11, orange 14, blue 14, matching the hand count.
