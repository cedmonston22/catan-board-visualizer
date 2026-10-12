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

## Scoring

The first end-to-end result: one photo in, the visible score per player out.

Visible score only: settlement 1, city 2, Longest Road 2. Largest Army and victory-point cards are not on the board.

Longest Road is a depth-first search with backtracking over each player's road graph (vertices = board vertices, edges = that player's roads):

- a trail can revisit a vertex but never reuse a road, so the search tracks used edges, not visited vertices
- an opponent's settlement or city ends a trail: it can reach that vertex but not continue through it
- the search starts from every vertex the player's roads touch and keeps the longest
- awarded only for 5 or more roads and a unique leader. A photo can't show who reached a tied length first, so a tie at the top awards it to nobody

Checks that passed: a line (3), a line cut by an opponent's settlement (2), the same line with the player's own settlement (3), a fork (3, not 4), and a triangle with a tail (4, which fails without backtracking).

| photo | result | check |
|---|---|---|
| test01 (unseen room) | blue 8 (Longest Road, 12), red 6, orange 6 | red's 11 roads form one unbroken line, so its longest road is 11. Blue has 14 roads but 2 sit on side branches, so its longest is 12 |
| round03_15 (endgame) | blue 8 (Longest Road, 7), orange 8, red 7, white 7 | 3 cities per player, matching the real board |
| IMG_8601, IMG_8602 (setup) | every player 2 points, longest road 1 | setup roads are separate, one per settlement |
| empty boards | no players | the robber has no color, so it never creates a player |

- `score_test01.jpg`, `score_round03_15.jpg`: original photo, the board read back with player colors, and the score panel.

Known limits: the detector's city vs settlement confusion changes a score by 1 per mistake, and tile/token classification is not built yet, so the regenerated board has no resources or numbers.

## Tiles and number tokens

Two classifiers, not one: numbers are placed randomly on tiles, so a combined "resource + number" class (about 55 combinations) would have no examples for most pairs. One model reads the terrain from the whole hex, the other reads the digit from the token.

### Labels from 3 boards, applied to 57 photos

- Labeled once per round on the empty-board shot (3 boards x 19 hexes = 57 labels), checked against the fixed Catan counts and hex by hex against the photos. `tile_labels_round1.jpg`.
- Rounds were found automatically by comparing each photo's 19 hex colors against the empty boards. Raw colors failed for Round 2 because the lighting changed halfway through it; normalizing each board by its own mean and spread fixed that.
- Every photo is rotated by some multiple of 60 degrees relative to its empty board (it depends on where the photo was taken from). The rotation is found by trying all 6 and keeping the one whose hex colors match best (median over hexes, so pieces on a few tiles don't matter). All 57 matched; worst distance 0.72 vs 1.3+ between different rounds. `rotation_alignment_check.jpg`.
- Result: 1,083 hex crops and 996 token crops. Round 3 held out for validation.

### Tiles: 100% on Round 3 (first try)

- YOLO11n-cls, 224 px. 100% on Round 3 by epoch 6. `tile_crops_by_resource.jpg`, `tiles_v1_training_curves.png`.

### Tokens: shortcut learning, then 100%

| version | input | Round 3 accuracy | what went wrong |
|---|---|---|---|
| v1 | 120 px square around the hex center, terrain visible | about 80%, the 12 got 0 of 15 | learned the terrain under the token instead of the digit |
| v2 | token found by circle detection, everything outside it painted gray, no mirror augmentation | **100% (329/329)** | nothing on Round 3 |

How v1 failed: with only 2 training layouts, each number sat on one or two specific tiles, and the terrain was easier to learn than the digit. Every big mistake matches a training-round pairing:

| Round 3 truth | v1 predicted | tile under it in Round 3 | why |
|---|---|---|---|
| 12 | 4 (10 times) | brick | Round 1's brick has a 4 |
| 11 | 10 (9 times) | wheat | Rounds 1 and 2 both have wheat 10 |
| 9 | 5 (9 times) | ore | Round 1 has ore 5 |
| 6 | 8 (7 times) | sheep, ore | Round 2 has sheep 8 and ore 8 |

v1's validation loss rose from epoch 4 while training loss kept falling. v2's both fall together.

The fix (`pipeline/tokens.py`):

- `cv2.HoughCircles` finds the token. Every token is the same size on the rectified board, so the radius is limited to 23-32 px; allowing 22-45 picked up larger false circles. `token_circle_detection.jpg`.
- The crop is recentered on the token and everything outside the circle is set to flat gray, so the only thing left to learn from is the digit and its dots.
- Training crops where no token was found (41, about 4%) are skipped rather than kept as terrain-only images labeled with a number.
- `fliplr=0.0`: a mirrored digit is not a real number.

`tokens_before_after_isolation.jpg`, `tokens_v1_training_curves.png`, `tokens_v2_training_curves.png`.

Known weakness: the circle finder occasionally locks onto a round shape in the terrain instead of the token (one such crop is visible in the 12 row of the before/after image). The model still scored 100%, but more layouts would make this rarer to matter.
