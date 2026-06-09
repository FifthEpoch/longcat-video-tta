# Per-video TTA-gain analysis  (baseline = NOTTA)

- Series: `sweep_experiment/results/panda_1000v_standard`
- TinyLoRA series: `delta_experiment/results/tinylora_panda_1000v_standard`
- Dynamicness JSON: `datasets/panda_1000_480p/dynamic_degree.json` (flow key = `mean_flow`)
- Captions CSV: `datasets/panda_1000_480p/metadata.csv`
- Methods analysed: NOTTA, ADA, ADA_NOPROMPT, LORA_R8_TTA, LORA_R8_TTA_NOPROMPT, TL_BARE_R2, TL_TIED_R2  (non-baseline: ADA, ADA_NOPROMPT, LORA_R8_TTA, LORA_R8_TTA_NOPROMPT, TL_BARE_R2, TL_TIED_R2)
- Common video_id intersection (across baseline + every method): **999**

## Data integrity

- Intersection rows missing `mean_flow`: **0** of 999
- Intersection rows missing caption: **0** of 999

| method | NaN-PSNR rows dropped before intersection |
|---|---:|
| `NOTTA` | 0 |
| `ADA` | 0 |
| `ADA_NOPROMPT` | 0 |
| `LORA_R8_TTA` | 0 |
| `LORA_R8_TTA_NOPROMPT` | 0 |
| `TL_BARE_R2` | 0 |
| `TL_TIED_R2` | 0 |

## Per-method ΔPSNR tail counts

Interpretation: large |Δ| tails mean TTA has real per-video effects even when the population mean is ≈ 0. A symmetric spread implies wins are paid for by equal-sized losses; a right-skewed spread means TTA is a net positive on a subset.

| method | N | mean Δ | median Δ | Δ>+1.0 | \|Δ\|≤1.0 | Δ<−1.0 | Δ>+0.5 | \|Δ\|≤0.5 | Δ<−0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `ADA` | 999 | +0.0080 | -0.0003 | 50 | 899 | 50 | 90 | 808 | 101 |
| `ADA_NOPROMPT` | 999 | +0.0020 | +0.0040 | 47 | 902 | 50 | 88 | 816 | 95 |
| `LORA_R8_TTA` | 999 | -0.0756 | -0.0109 | 7 | 971 | 21 | 17 | 943 | 39 |
| `LORA_R8_TTA_NOPROMPT` | 999 | -0.0650 | -0.0078 | 12 | 967 | 20 | 19 | 931 | 49 |
| `TL_BARE_R2` | 999 | +0.0108 | -0.0016 | 9 | 982 | 8 | 29 | 950 | 20 |
| `TL_TIED_R2` | 999 | +0.0027 | +0.0009 | 9 | 980 | 10 | 27 | 951 | 21 |

## Correlation between ΔPSNR and per-video features

Reported as Pearson r (Spearman ρ). Both use the intersection of finite ΔPSNR with finite feature; per-method N may differ slightly when individual rows have NaN features.

| method | r(Δ, mean_flow) ρ | r(Δ, baseline PSNR) ρ | r(Δ, caption words) ρ |
|---|---|---|---|
| `ADA` | -0.097 (-0.069) | -0.004 (+0.013) | -0.010 (+0.012) |
| `ADA_NOPROMPT` | -0.070 (-0.085) | -0.045 (+0.006) | -0.018 (-0.025) |
| `LORA_R8_TTA` | +0.021 (+0.073) | -0.143 (-0.088) | +0.001 (-0.024) |
| `LORA_R8_TTA_NOPROMPT` | -0.005 (+0.036) | -0.112 (-0.045) | -0.008 (-0.017) |
| `TL_BARE_R2` | -0.046 (-0.010) | -0.017 (+0.011) | -0.005 (-0.018) |
| `TL_TIED_R2` | -0.058 (+0.014) | -0.062 (-0.074) | +0.053 (+0.062) |

## Top 10 winners / losers per method

Rank by ΔPSNR. Caption truncated to 80 chars; mean_flow / baseline PSNR pulled from this video's row in the intersection.

### `ADA` — top 10 winners

| # | ΔPSNR | baseline PSNR | mean_flow | video_id | caption |
|---:|---:|---:|---:|---|---|
| 1 | +8.669 | 13.52 | 0.132 | `panda_0867` | ['A man is holding a fishing lure and explaining how to use it.', 'A person is … |
| 2 | +4.641 | 17.36 | 0.079 | `panda_0210` | ['An aerial view of a snowy mountain range with snow on the ground.', 'A person… |
| 3 | +4.638 | 14.10 | 0.572 | `panda_0941` | ['A person is signing a form with a pen.', 'Cars are driving on a highway with … |
| 4 | +4.530 | 33.84 | 4.091 | `panda_0302` | ['The screen of a smartphone in hindi.', 'A mobile phone with a message in hind… |
| 5 | +4.202 | 23.05 | 0.102 | `panda_0729` | ['There are two men sitting on red chairs in a room with shelves filled with bo… |
| 6 | +3.865 | 41.01 | 0.103 | `panda_0013` | ['There is a small spot on the ceiling of a bathroom caused by an accumulation … |
| 7 | +3.571 | 14.23 | 0.116 | `panda_0675` | ['There are people skating on an outdoor ice rink in front of a large building … |
| 8 | +3.566 | 21.06 | 0.576 | `panda_0090` | ['A building in a minecraft game.', 'A room in a minecraft game.', 'A minecraft… |
| 9 | +3.496 | 14.04 | 0.071 | `panda_0461` | ['An iphone, a cup of coffee, a yellow sticky note, and a computer are on a des… |
| 10 | +3.388 | 8.30 | 0.117 | `panda_0287` | ['An essay with the words thesis statement.', 'A page of a paper with the title… |

### `ADA` — top 10 losers

| # | ΔPSNR | baseline PSNR | mean_flow | video_id | caption |
|---:|---:|---:|---:|---|---|
| 1 | -6.371 | 14.57 | 0.517 | `panda_0942` | ['The notre dame cathedral in paris is on fire and smoke is billowing out of th… |
| 2 | -5.717 | 16.68 | 0.500 | `panda_0774` | ['A blue background with the words miniature float challenge from my scrap room… |
| 3 | -5.118 | 30.72 | 1.339 | `panda_0576` | ['The text zero hero appears on a red background with light flashes.', 'A paint… |
| 4 | -4.435 | 12.42 | 0.072 | `panda_0396` | ['A wooden spoon sitting on top of a black lid.', 'The object is a black metal … |
| 5 | -4.064 | 23.57 | 15.732 | `panda_0365` | ['A bmw car driving down a road with a grassy field in the background.', 'A sil… |
| 6 | -3.631 | 17.75 | 0.156 | `panda_0452` | ['The computer screen is displaying a white screen with a black border, and the… |
| 7 | -3.463 | 29.56 | 29.716 | `panda_0089` | ['An intel logo on a blue background.', 'An intel logo on a blue background.', … |
| 8 | -2.965 | 34.97 | 0.113 | `panda_0720` | ['A person is holding a smartphone with a picture of a game on the screen.', 'A… |
| 9 | -2.917 | 20.17 | 0.254 | `panda_0165` | ['Two men standing next to each other in a stadium.', 'A man is talking to anot… |
| 10 | -2.863 | 13.79 | 0.099 | `panda_0225` | ['A stick figure in a blue hoodie and glasses is waving its arms up and down on… |

### `ADA_NOPROMPT` — top 10 winners

| # | ΔPSNR | baseline PSNR | mean_flow | video_id | caption |
|---:|---:|---:|---:|---|---|
| 1 | +8.922 | 14.04 | 0.071 | `panda_0461` | ['An iphone, a cup of coffee, a yellow sticky note, and a computer are on a des… |
| 2 | +8.587 | 13.52 | 0.132 | `panda_0867` | ['A man is holding a fishing lure and explaining how to use it.', 'A person is … |
| 3 | +5.661 | 22.86 | 1.600 | `panda_0925` | ['A computer screen showing a facebook page.', 'A screenshot of a computer scre… |
| 4 | +4.942 | 10.93 | 0.442 | `panda_0878` | ['A blue background with the words spice cabinet and french style spice jars.',… |
| 5 | +4.606 | 33.84 | 4.091 | `panda_0302` | ['The screen of a smartphone in hindi.', 'A mobile phone with a message in hind… |
| 6 | +4.371 | 17.36 | 0.079 | `panda_0210` | ['An aerial view of a snowy mountain range with snow on the ground.', 'A person… |
| 7 | +4.238 | 23.05 | 0.102 | `panda_0729` | ['There are two men sitting on red chairs in a room with shelves filled with bo… |
| 8 | +3.862 | 18.98 | 0.203 | `panda_0860` | ['A poster for the movie the order 1886.', "The characters in the game assassin… |
| 9 | +3.855 | 41.01 | 0.103 | `panda_0013` | ['There is a small spot on the ceiling of a bathroom caused by an accumulation … |
| 10 | +3.634 | 8.30 | 0.117 | `panda_0287` | ['An essay with the words thesis statement.', 'A page of a paper with the title… |

### `ADA_NOPROMPT` — top 10 losers

| # | ΔPSNR | baseline PSNR | mean_flow | video_id | caption |
|---:|---:|---:|---:|---|---|
| 1 | -9.108 | 28.34 | 1.388 | `panda_0192` | ['A blue bowl of food with pasta, tomatoes, and other vegetables on a table.', … |
| 2 | -8.828 | 35.02 | 0.150 | `panda_0074` | ['A diagram showing the structure of a neuron.', 'It is a video about the nervo… |
| 3 | -6.218 | 14.57 | 0.517 | `panda_0942` | ['The notre dame cathedral in paris is on fire and smoke is billowing out of th… |
| 4 | -5.979 | 30.72 | 1.339 | `panda_0576` | ['The text zero hero appears on a red background with light flashes.', 'A paint… |
| 5 | -5.633 | 16.68 | 0.500 | `panda_0774` | ['A blue background with the words miniature float challenge from my scrap room… |
| 6 | -5.387 | 19.26 | 0.328 | `panda_0644` | ['A man with glasses sitting in front of a light.', 'A man in a lab coat lookin… |
| 7 | -4.287 | 12.42 | 0.072 | `panda_0396` | ['A wooden spoon sitting on top of a black lid.', 'The object is a black metal … |
| 8 | -3.340 | 29.56 | 29.716 | `panda_0089` | ['An intel logo on a blue background.', 'An intel logo on a blue background.', … |
| 9 | -3.165 | 34.97 | 0.113 | `panda_0720` | ['A person is holding a smartphone with a picture of a game on the screen.', 'A… |
| 10 | -3.083 | 20.54 | 3.408 | `panda_0104` | ['A man wearing headphones is standing in front of a black background.', 'There… |

### `LORA_R8_TTA` — top 10 winners

| # | ΔPSNR | baseline PSNR | mean_flow | video_id | caption |
|---:|---:|---:|---:|---|---|
| 1 | +8.254 | 14.04 | 0.071 | `panda_0461` | ['An iphone, a cup of coffee, a yellow sticky note, and a computer are on a des… |
| 2 | +7.408 | 10.28 | 1.258 | `panda_0862` | ['A group of cartoon people with their arms up in the air.', 'A dragon ball z c… |
| 3 | +3.130 | 7.82 | 0.366 | `panda_0555` | ['A cartoon girl looking at her phone with a speech bubble that says good morni… |
| 4 | +2.389 | 10.94 | 1.636 | `panda_0236` | ['A poster with the words using minor pentatonic over major 7th chords.', 'The … |
| 5 | +1.387 | 40.33 | 0.403 | `panda_0928` | ['It is a black background with the words "headliner install part 1" written in… |
| 6 | +1.142 | 12.70 | 4.003 | `panda_0312` | ['A man is standing next to an armored vehicle in a workshop.', 'A mechanic is … |
| 7 | +1.098 | 37.75 | 2.533 | `panda_0641` | ['A video game screen with text on it.', 'A screenshot of a video game showing … |
| 8 | +0.968 | 9.44 | 0.404 | `panda_0693` | ["A close up of a person's eye.", 'A black and white cat is sitting in front of… |
| 9 | +0.868 | 18.25 | 0.065 | `panda_0086` | ['A video game showing a girl in a pink dress.', 'A room in the sims game.', 'A… |
| 10 | +0.766 | 23.21 | 0.797 | `panda_0732` | ['The mugshots of five men who have been arrested.', 'An aerial view of a dam t… |

### `LORA_R8_TTA` — top 10 losers

| # | ΔPSNR | baseline PSNR | mean_flow | video_id | caption |
|---:|---:|---:|---:|---|---|
| 1 | -22.396 | 44.55 | 0.271 | `panda_0098` | ['The words "home workshop makeover tour" are written on a white background.', … |
| 2 | -7.296 | 29.04 | 0.633 | `panda_0255` | ['The text "hayward\'s fitness bodybuilding" is written on a black background.'… |
| 3 | -4.194 | 21.42 | 2.608 | `panda_0151` | ['A woman pouring liquid into a pot on a stove.', 'A woman in a kitchen holding… |
| 4 | -3.770 | 16.68 | 0.500 | `panda_0774` | ['A blue background with the words miniature float challenge from my scrap room… |
| 5 | -3.488 | 22.86 | 1.600 | `panda_0925` | ['A computer screen showing a facebook page.', 'A screenshot of a computer scre… |
| 6 | -2.482 | 35.27 | 2.277 | `panda_0335` | ['A pink background with streaks of light.', 'The anime character is standing i… |
| 7 | -1.844 | 16.31 | 0.083 | `panda_0677` | ['A yellow school bus is driving down a street next to trees.', 'A school bus i… |
| 8 | -1.826 | 27.39 | 1.154 | `panda_0453` | ['A mobile phone with the word subscribe written in hindi on a yellow and blue … |
| 9 | -1.628 | 24.48 | 4.516 | `panda_0811` | ['A man and a woman sitting in a chair talking.', 'The man is wearing a black j… |
| 10 | -1.459 | 20.19 | 0.082 | `panda_0241` | ['A soccer stadium with a person on a screen.', 'A video game screen showing a … |

### `LORA_R8_TTA_NOPROMPT` — top 10 winners

| # | ΔPSNR | baseline PSNR | mean_flow | video_id | caption |
|---:|---:|---:|---:|---|---|
| 1 | +7.942 | 14.04 | 0.071 | `panda_0461` | ['An iphone, a cup of coffee, a yellow sticky note, and a computer are on a des… |
| 2 | +7.554 | 10.28 | 1.258 | `panda_0862` | ['A group of cartoon people with their arms up in the air.', 'A dragon ball z c… |
| 3 | +3.161 | 7.82 | 0.366 | `panda_0555` | ['A cartoon girl looking at her phone with a speech bubble that says good morni… |
| 4 | +2.826 | 31.13 | 0.593 | `panda_0431` | ['A black background with red text on it.', 'A black and red background with th… |
| 5 | +1.691 | 14.89 | 0.219 | `panda_0012` | ['A cartoon girl with a smile on her face.', 'A person is cooking food in a pot… |
| 6 | +1.497 | 9.58 | 0.573 | `panda_0584` | ['There is a man standing in front of a booth displaying bicycles, and he is ta… |
| 7 | +1.443 | 23.64 | 3.446 | `panda_0618` | ['Two sports cars parked in front of a garage.', 'A collage of many different c… |
| 8 | +1.384 | 40.33 | 0.403 | `panda_0928` | ['It is a black background with the words "headliner install part 1" written in… |
| 9 | +1.366 | 25.05 | 5.883 | `panda_0888` | ['A man with a rope climbing into an ice cave.', 'A computer monitor with a pic… |
| 10 | +1.253 | 27.25 | 0.058 | `panda_0650` | ['A small green train is on a toy track.', 'A toy thomas the tank engine next t… |

### `LORA_R8_TTA_NOPROMPT` — top 10 losers

| # | ΔPSNR | baseline PSNR | mean_flow | video_id | caption |
|---:|---:|---:|---:|---|---|
| 1 | -23.516 | 44.55 | 0.271 | `panda_0098` | ['The words "home workshop makeover tour" are written on a white background.', … |
| 2 | -5.574 | 16.68 | 0.500 | `panda_0774` | ['A blue background with the words miniature float challenge from my scrap room… |
| 3 | -4.133 | 21.42 | 2.608 | `panda_0151` | ['A woman pouring liquid into a pot on a stove.', 'A woman in a kitchen holding… |
| 4 | -3.712 | 29.56 | 29.716 | `panda_0089` | ['An intel logo on a blue background.', 'An intel logo on a blue background.', … |
| 5 | -2.821 | 20.54 | 3.408 | `panda_0104` | ['A man wearing headphones is standing in front of a black background.', 'There… |
| 6 | -2.503 | 20.17 | 0.254 | `panda_0165` | ['Two men standing next to each other in a stadium.', 'A man is talking to anot… |
| 7 | -2.118 | 13.79 | 0.099 | `panda_0225` | ['A stick figure in a blue hoodie and glasses is waving its arms up and down on… |
| 8 | -1.918 | 24.60 | 0.072 | `panda_0246` | ['A computer screen showing an image of a tank.', 'A screenshot of a game showi… |
| 9 | -1.625 | 8.48 | 0.198 | `panda_0254` | ['A screen shot of a menu in a video game.', 'The player is playing a first-per… |
| 10 | -1.543 | 15.76 | 0.678 | `panda_0234` | ['The product is a packet of quinoa flakes.', 'A blender sitting on top of a wo… |

### `TL_BARE_R2` — top 10 winners

| # | ΔPSNR | baseline PSNR | mean_flow | video_id | caption |
|---:|---:|---:|---:|---|---|
| 1 | +7.530 | 10.28 | 1.258 | `panda_0862` | ['A group of cartoon people with their arms up in the air.', 'A dragon ball z c… |
| 2 | +3.143 | 7.82 | 0.366 | `panda_0555` | ['A cartoon girl looking at her phone with a speech bubble that says good morni… |
| 3 | +2.886 | 31.13 | 0.593 | `panda_0431` | ['A black background with red text on it.', 'A black and red background with th… |
| 4 | +1.656 | 27.25 | 0.058 | `panda_0650` | ['A small green train is on a toy track.', 'A toy thomas the tank engine next t… |
| 5 | +1.531 | 40.33 | 0.403 | `panda_0928` | ['It is a black background with the words "headliner install part 1" written in… |
| 6 | +1.405 | 23.64 | 3.446 | `panda_0618` | ['Two sports cars parked in front of a garage.', 'A collage of many different c… |
| 7 | +1.256 | 13.31 | 1.740 | `panda_0482` | ['A black man is standing in front of a dark background, wearing a plaid shirt … |
| 8 | +1.235 | 37.75 | 2.533 | `panda_0641` | ['A video game screen with text on it.', 'A screenshot of a video game showing … |
| 9 | +1.023 | 14.10 | 0.572 | `panda_0941` | ['A person is signing a form with a pen.', 'Cars are driving on a highway with … |
| 10 | +0.965 | 16.02 | 0.155 | `panda_0031` | ['A view of a train track next to a fence.', 'A blue and silver passenger train… |

### `TL_BARE_R2` — top 10 losers

| # | ΔPSNR | baseline PSNR | mean_flow | video_id | caption |
|---:|---:|---:|---:|---|---|
| 1 | -3.736 | 16.68 | 0.500 | `panda_0774` | ['A blue background with the words miniature float challenge from my scrap room… |
| 2 | -3.664 | 29.56 | 29.716 | `panda_0089` | ['An intel logo on a blue background.', 'An intel logo on a blue background.', … |
| 3 | -1.646 | 23.22 | 0.339 | `panda_0522` | ['The person is holding markers on a black cardboard.', 'A person writing with … |
| 4 | -1.625 | 20.19 | 0.082 | `panda_0241` | ['A soccer stadium with a person on a screen.', 'A video game screen showing a … |
| 5 | -1.452 | 10.94 | 1.636 | `panda_0236` | ['A poster with the words using minor pentatonic over major 7th chords.', 'The … |
| 6 | -1.305 | 16.31 | 0.083 | `panda_0677` | ['A yellow school bus is driving down a street next to trees.', 'A school bus i… |
| 7 | -1.227 | 14.04 | 0.071 | `panda_0461` | ['An iphone, a cup of coffee, a yellow sticky note, and a computer are on a des… |
| 8 | -1.184 | 24.60 | 0.072 | `panda_0246` | ['A computer screen showing an image of a tank.', 'A screenshot of a game showi… |
| 9 | -0.923 | 19.96 | 0.316 | `panda_0791` | ['A man with long hair and beard is sitting on a sofa and talking to the camera… |
| 10 | -0.899 | 35.27 | 2.277 | `panda_0335` | ['A pink background with streaks of light.', 'The anime character is standing i… |

### `TL_TIED_R2` — top 10 winners

| # | ΔPSNR | baseline PSNR | mean_flow | video_id | caption |
|---:|---:|---:|---:|---|---|
| 1 | +3.151 | 7.82 | 0.366 | `panda_0555` | ['A cartoon girl looking at her phone with a speech bubble that says good morni… |
| 2 | +2.893 | 31.13 | 0.593 | `panda_0431` | ['A black background with red text on it.', 'A black and red background with th… |
| 3 | +2.423 | 23.21 | 0.797 | `panda_0732` | ['The mugshots of five men who have been arrested.', 'An aerial view of a dam t… |
| 4 | +1.606 | 23.64 | 3.446 | `panda_0618` | ['Two sports cars parked in front of a garage.', 'A collage of many different c… |
| 5 | +1.575 | 9.58 | 0.573 | `panda_0584` | ['There is a man standing in front of a booth displaying bicycles, and he is ta… |
| 6 | +1.477 | 21.62 | 4.589 | `panda_0591` | ['A green tent set up on a grassy field.', 'A close up of a yellow cord attache… |
| 7 | +1.414 | 23.05 | 0.102 | `panda_0729` | ['There are two men sitting on red chairs in a room with shelves filled with bo… |
| 8 | +1.293 | 13.31 | 1.740 | `panda_0482` | ['A black man is standing in front of a dark background, wearing a plaid shirt … |
| 9 | +1.029 | 12.70 | 4.003 | `panda_0312` | ['A man is standing next to an armored vehicle in a workshop.', 'A mechanic is … |
| 10 | +0.985 | 13.15 | 5.773 | `panda_0851` | ['A video game with two characters fighting each other.', 'A video game with tw… |

### `TL_TIED_R2` — top 10 losers

| # | ΔPSNR | baseline PSNR | mean_flow | video_id | caption |
|---:|---:|---:|---:|---|---|
| 1 | -4.096 | 21.42 | 2.608 | `panda_0151` | ['A woman pouring liquid into a pot on a stove.', 'A woman in a kitchen holding… |
| 2 | -3.890 | 29.56 | 29.716 | `panda_0089` | ['An intel logo on a blue background.', 'An intel logo on a blue background.', … |
| 3 | -1.968 | 37.75 | 2.533 | `panda_0641` | ['A video game screen with text on it.', 'A screenshot of a video game showing … |
| 4 | -1.794 | 10.94 | 1.636 | `panda_0236` | ['A poster with the words using minor pentatonic over major 7th chords.', 'The … |
| 5 | -1.786 | 16.31 | 0.083 | `panda_0677` | ['A yellow school bus is driving down a street next to trees.', 'A school bus i… |
| 6 | -1.702 | 23.22 | 0.339 | `panda_0522` | ['The person is holding markers on a black cardboard.', 'A person writing with … |
| 7 | -1.426 | 30.40 | 1.393 | `panda_0153` | ['A person holding an iphone in their hand.', 'There is heavy traffic on a high… |
| 8 | -1.201 | 20.19 | 0.082 | `panda_0241` | ['A soccer stadium with a person on a screen.', 'A video game screen showing a … |
| 9 | -1.164 | 14.10 | 0.572 | `panda_0941` | ['A person is signing a form with a pen.', 'Cars are driving on a highway with … |
| 10 | -1.052 | 35.27 | 2.277 | `panda_0335` | ['A pink background with streaks of light.', 'The anime character is standing i… |
