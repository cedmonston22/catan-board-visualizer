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

## Regenerated board

Goal 2 of the original plan: one photo in, a clean digital board out (resources, numbers, robber, every player's pieces in their colors, and the score panel). `pipeline/hexes.py` reads the 19 hexes, `pipeline/render.py` draws the board from the board state alone, with no photo pixels.

`render_test01_photo_to_board.jpg`: the unseen-room photo and its regenerated board.

### Reading numbers in real photos

Three problems showed up once the token reader ran on whole photos instead of training crops, each fixed with a rule rather than more training:

| problem | fix | effect |
|---|---|---|
| when no token was found, the model guessed on a terrain-only crop (most often reading a 5 as a 2) | no token found means the number is unknown, not guessed | 11 wrong numbers became honest unknowns |
| some tokens sit near the edge of their hex, outside the 40 px search area | search the whole hex (up to 70 px off-center), keeping the 23-32 px radius limit | tokens not found: 41 to 0, and the 38 recovered were read correctly |
| the robber covers a token, and the model still guessed a number | the snapped robber's hex is marked unknown | test01's covered token stopped reading as a third 9 |

Then deduction (`fill_unknown_numbers`): every board has exactly one 2, one 12, and two of every other number. Whatever is missing from the numbers that were read must be the hidden one. It only fills in when exactly one hex is unknown and exactly one number is left over, and it backs off if any number was read too many times. On test01, the board showed one 4 where there should be two, so the robber's token is the 4.

Results across the 57 labeled photos:

| | count |
|---|---|
| resources right | 1,083 / 1,083 |
| numbers read right by the model | 978 |
| numbers filled right by deduction | **43 / 43** (no wrong fills) |
| still unknown | 3 (photos with two unknowns, where filling would be a guess) |
| wrong | 2 |
| **numbers right overall** | **1,021 / 1,026 (99.5%)** |

test01 (a layout the classifiers never saw): 19 / 19 resources, all 17 visible numbers right, and the hidden one deduced correctly.

## Label errors found with a game rule

A user-reported bug: in IMG_8616 a settlement rendered blue, but the piece in the photo is orange.

The color model was not misreading it. Its crop was labeled blue in `data/colors_all.json`: an automatic pre-label, fooled by the sea filling half the crop, that slipped through review. IMG_8616 is a training photo, so the classifier learned the mistake and was 99.8% confident the orange piece was blue. The usual audit (model vs labels) could not see it, because the model and the label agreed.

What found it was a rule from the game that does not depend on any labels: a settlement or city should touch at least one road of its own color. Checking every building on every photo flagged three:

| photo | piece | problem |
|---|---|---|
| IMG_8616 | settlement | orange, labeled blue: label wrong, model learned it |
| IMG_8633 | settlement | white under the warm Round 2 lamp, labeled orange: label wrong |
| IMG_8628 | road | red with glare, model said orange: model wrong, label right |

`label_fix_suspect_pieces.jpg` shows each flagged piece with its prediction and label.

Fix: corrected the two labels and retrained (`colors_v2`, 100% on Round 3). After that, the rule check finds nothing on any of the 58 photos, the label audit finds no mismatches, and the IMG_8628 road was fixed too, even though its own label was always right. `label_fix_IMG_8616_before_after.jpg`.

Takeaways:

- A high accuracy score only measures agreement with the labels. A model trained on a wrong label will confidently reproduce it, and an audit against those same labels will pass.
- Both bad labels were automatic pre-labels. Review catches most of these, but not all, especially when the background resembles another player's color (sea vs blue, warm light vs orange).
- Checks that come from the domain, like the road rule here or the token counts in deduction, catch errors that the data alone can't. The road rule is a good candidate for a permanent check that flags suspicious pieces in the pipeline.
