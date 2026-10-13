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

### Tested, not adopted (later adopted): rescuing low-confidence roads

test01 is missing one road: the detector saw it at only 14% confidence (below the 25% cutoff) because a settlement hides half of it, in a room the detector never trained on.

- Lowering the cutoff globally recovers nothing on the 57 labeled photos (the same 2 pieces are missed at every cutoff from 25% down to 5%) while false pieces rise from 7 to 14.
- A targeted rule was tested instead: accept a 10-25% road only if its edge is empty and both ends touch a piece of the same color. It recovers exactly the test01 road and never fires on the 57 labeled photos (0 false pieces).
- Not adopted yet: it is safe, but one rescued road is not enough evidence that it matters. Worth revisiting once there are more photos from new rooms.
- Adopted after the unseen test showed the same failure again; see "Road rescue" at the end.

## Automatic corners and read_board

The last manual step was clicking 6 corners per photo. The product goal is to take a photo and get the board and scores, so corners have to be found automatically.

### Corner detector

- The 348 clicked corners (58 photos x 6) became training labels for free: each click is a small box, sized at 25% of the distance between neighboring corners so it scales with the board. `corner_training_boxes.jpg`.
- YOLO11s, imgsz 1024, 39 training photos. Round 3: precision 0.92, recall 0.84, mAP50 0.91. mAP50-95 is only 0.49, but only the box centers matter here.

### Choosing the 6 corners

| step | Round 3 photos accepted | notes |
|---|---|---|
| take the 6 most confident boxes | 7 / 18 | duplicates: two boxes on one corner, another corner missed |
| remove duplicates, try every combination of 6, keep the most board-shaped | 14 / 18 | the rest have one corner hidden or missed |
| also complete a hidden 6th corner from 5 found ones | 17 / 18 | **rejected**: some "passing" boards were 400 px wrong |
| completion, but only accepted if the rectified board passes the tile check | 17 / 18 | safe |

Completing from 5 corners failed at first because a homography has 8 unknowns and 5 points give 10 equations, so almost any 5 points fit almost perfectly. The fit error said nothing.

The tile check: rectify with the candidate corners, classify the 19 hexes, and accept only if the counts are exactly 4 wood, 4 sheep, 4 wheat, 3 brick, 3 ore, 1 desert with average confidence of at least 80%. Wrong corners turn sea and table edges into "tiles", which fail the counts or the confidence. It is the same pattern as the other fixes: check the answer against the rules of the game.

### End to end: automatic vs clicked corners, all 58 photos

| | photos |
|---|---|
| same scores as with clicked corners | 53 |
| different scores | 2 (test01, round03_13) |
| rejected, so the corners need to be placed by hand | 3 |

53 of the 55 accepted boards end with the right scores (96%). The 2 misses have one corner 60-90 px off: close enough for the tiles to pass, stretched enough to move a piece or misread a token. `auto_vs_clicked_corners_test01.jpg`: in the stretched top-right area, the brick 5 reads as a 12 and two red pieces fall off their spots. Deduction then sees two 12s and correctly refuses to guess the robber's number.

This is the corner detector's limit (39 photos from one table), and the main reason to collect new photos in different rooms next.

### read_board

`python -m pipeline.read_board photo.jpg`: any photo in, board and scores out, no corner file. `read_board_round03_15.jpg`.

- Built around one function, `read_board(image, models, board, corners=None)`, that works on an image in memory and returns a `BoardResult`: corners (automatic or manual), pieces, hexes, scores, the rendered board, and plain-language warnings.
- If the automatic corners fail the checks, it returns a problem instead of a wrong board. The command line then opens the click window for that photo; the future phone page will show draggable corners instead.
- The future web app calls the same function, so the phone and the command line give identical answers.

## New rounds: a new setting

Rounds 1-3 were all on one table, mostly in the evening. Rounds 4 and 5 (40 photos) were shot on carpet in a different room: Round 4 in cool, even light, Round 5 under a warm lamp, with very steep and very distant shots on purpose. Both went into training; Round 3 stayed the validation set so every before/after number below is on the same photos. `new_rounds_round05.jpg`.

### How the old models did in the new setting (before retraining)

| model | old setting (Round 3) | new setting (Rounds 4-5) |
|---|---|---|
| automatic corners | 17/18 accepted | **20/40** accepted. `new_rounds_v1_corner_results.jpg` |
| piece detector (catan_v2) | mAP50-95 0.85 | **0.71**: 41 city/settlement swaps, 36 roads missed, 128 false pieces |
| tile classifier (tiles_v1) | tile check passes 18/18 | **28/40**: warm, washed-out tiles read as desert |
| token reader (tokens_v2) | 99.7% of numbers | **88%** |
| color (colors_v2) | 0 mistakes | **12** mistakes, 10 of them white read as orange under the warm lamp |

Every model lost accuracy in a room it hadn't seen. This is the strongest evidence in the project that the original numbers were measured on the same conditions the models trained on.

### Labeling the new rounds

The existing models did most of the labeling:

- **Corners:** 20 of 40 found automatically, 20 clicked by hand.
- **Rounds and rotations:** both rounds matched their empty boards (Round 5's empty board was shot at the end of the game).
- **Tiles:** 2 boards x 19 hexes labeled by hand, about 4 minutes.
- **Pieces:** auto-labeled and reviewed in Roboflow.
- **Colors:** colors_v2 pre-labeled the 1,081 new crops; review corrected only 17 (the old k-means pre-labels needed 173 corrections on a similar amount). Reviewed labels now store their photo, piece type, and box, so they carry over to future exports by box overlap instead of by file name.

### After retraining

| model | Round 3 (held out) | new setting | notes |
|---|---|---|---|
| corners_v2 | 17/18, one corner over 30 px (was 4) | 39/40 automatic with tiles_v2 | test01's worst corner 92 px to 24 px |
| tiles_v2 | 100% | tile check 40/40 | the 12 rejections were the tile classifier, not the corners: with hand-clicked corners, tiles_v1 still failed them |
| catan_v3 | mAP50-95 0.865 (was 0.851), false pieces 21 to 12 | trained on these | |
| colors_v4 | 0 mistakes | 2 mistakes in about 3,200 crops overall | v2 had 17 |
| tokens_v3 | 98.8% (v2: 99.7%) | 95% (v2: 88%) | mixed; v2 kept until unseen test photos decide |

New-setting numbers for retrained models are on their own training photos, so they show what was learned, not generalization. The unseen test photos are the real test.

### More bad labels, all from warm light

The road rule (a building should touch a road of its own color) and a second rule (no player owns more than 5 settlements, 4 cities, or 15 roads) flagged 12 problems with colors_v2, mostly in Round 5. Retraining fixed most of them, but the new model then "disagreed" with 7 labels in IMG_8633, the warm-lamp half of Round 2. They were white pieces labeled orange: k-means pre-labels from the first round of labeling that had passed review. `warm_light_label_fix_IMG_8633.jpg`.

- That makes three times a model disagreement or a game rule exposed wrong labels, all automatic pre-labels, all in warm light. Warm-light photos need extra care in review.
- A first colors_v4 run learned nothing new: the labels were fixed but the dataset wasn't rebuilt, and the classifier reads labels from which folder a crop is in. Labels, then rebuild, then train.
- With corrected labels, colors_v4 has 0 mistakes on Rounds 1-3 and the road rule flags only one steep, distant photo, where a red road was probably not detected.

### Process lessons

- Never rebuild a dataset while a model is training on it: the builders wipe their folder, and the tile training crashed at epoch 9 when its images disappeared.
- Improving one model exposes the next weakest: better corners revealed that the tile classifier was the real reason boards were rejected.

## Unseen test photos

Four photos never used for training or validation, with the true score of every player written down before running anything. Run exactly as the app would: `read_board` with automatic corners, no manual help, with catan_v3, colors_v4, corners_v2, and tiles_v2.

| photo | result | what went wrong |
|---|---|---|
| test02 | blue 3 (should be 2), white 1 (should be 2) | a notch in the blue sea frame detected as a settlement at 29% confidence and colored blue, and one white settlement missed |
| test03 | all 4 players right | |
| test04 | all 4 players right | |
| test05 | red 5 points (should be 6) | a red city read as a settlement |

- **2 of 4 photos fully right, 14 of 16 player scores right.** Longest Road (nobody, in all 4) right every time.
- **Automatic corners: 4 / 4.** Before Rounds 4-5, half the photos from this room were rejected.
- **Colors: no mistakes** on any piece the detector found.
- **Pieces are the remaining weak spot:** all 3 mistakes are detection (one phantom piece, one missed piece, one city/settlement swap).
- Number reading was not scored: there is no tile-layout answer key for these photos, so tokens_v2 vs v3 is still open.

How much to trust it: 4 photos is a small sample, all in the Round 4 room (new layouts and game states, but a familiar setting), and two share a layout. Neither error was tuned away: raising the confidence cutoff would remove the phantom piece, but that is fitting to one photo, and the earlier cutoff sweep showed it trades false pieces for missed ones.

`unseen_test04.jpg`: a fully correct photo and its regenerated board. `unseen_test_errors.jpg`: the pieces behind the two wrong scores.

## Road rescue: roads hidden behind buildings

test04's scores were right, but its board was not: a white road on the left coast, mostly hidden behind a white settlement, was missing. The detector saw it at 24% confidence, just under the 25% cutoff. It is the same failure as test01's road (14%, also behind a settlement), so the rule tested earlier was adopted.

The rule, in `snap.py`: the detector runs once at 10%. Everything at 25% or more is used as before. A road between 10% and 25% is kept only if its edge is empty and both of its ends touch a piece of the same color (a building on that corner, or a road ending there). Weak settlements, cities, and robbers are never rescued. Ends are checked only against confident pieces, so two weak roads can't vouch for each other.

| set | weak road candidates | rescued | real (matches a labeled road) | fake |
|---|---|---|---|---|
| 98 photos with clicked corners | 116 | 2 | 2 | 0 |
| test01-05, automatic corners | 8 | 1 (test04 white) | 1 | 0 |

- The two rescues on labeled photos (round04_06 orange, round05_08 red, both 16%) are real roads the confident pass missed. These are training photos, so they are weaker evidence than test04.
- Scores are unchanged on all 5 test photos and nobody gains Longest Road. test02 and test05 keep their building errors, which a road rule can't fix.
- test01's road no longer needs rescuing: catan_v3 finds it above the cutoff.

### Why both ends, not one

The game only requires a road to touch its own color at one end, so the simpler rule was measured too:

| rule | rescued | real | fake |
|---|---|---|---|
| both ends (adopted) | 2 | 2 | 0 |
| one end (the game rule) | 4 | 2 | 2 (IMG_8623 blue, round04_11 white) |

One end found no extra real roads and added two fake ones. The rule is not asking "would a road be legal here" but "is this weak box really a road". Every settlement has open edges next to it, so a blurry box beside one passes the one-end check easily. A road hidden behind a building is almost always in the middle of a chain, with the building on one side and the player's next road on the other. The cost: a hidden road at the very end of a chain won't be rescued. That hasn't happened yet, and a missing road is cheaper than a fake one, which can change Longest Road.

`road_rescue_test04.jpg`: the hidden road in the photo, the board before, and the board after.
