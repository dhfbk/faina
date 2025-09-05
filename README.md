# Fine-grained Fallacy Detection with Human Label Variation

This repository contains materials associated to the paper:

Alan Ramponi, Agnese Daffara, and Sara Tonelli. 2025. **Fine-grained Fallacy Detection with Human Label Variation**. In *Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)*, pages 762–784, Albuquerque, New Mexico. Association for Computational Linguistics. [[cite]](#citation) [[paper]](https://aclanthology.org/2025.naacl-long.34/)

- :page_with_curl: [**Faina dataset**](#faina-dataset)
- :rocket: [**Fallacy detection models**](#fallacy-detection-models)
- :book: [**Further information**](#further-information)
- :pencil2: [**Citation**](#citation)


✨**NEWS**✨ – We are organizing **FadeIT**, a shared task on fallacy detection based on this work (register [here](https://sites.google.com/fbk.eu/fadeit2026))!


## Faina dataset

Faina is the first dataset for fine-grained fallacy detection that embraces multiple plausible answers and natural disagreement. Faina includes over 11K span-level annotations with overlaps across 20 fallacy types on social media posts in Italian (covering four full years) about migration, climate change, and public health given by two expert annotators. Through an extensive annotation study that allowed discussion over multiple rounds, we minimize annotation errors whilst keeping signals of human label variation.

> :page_with_curl: **Dataset request**: please write an e-mail to the first/corresponding author to request the Faina dataset. The dataset can be used for non-commercial research purposes only and the user must declare to avoid the redistribution of the dataset to third parties or in online repositories, deanonymization (by any means), and to exclude data misuse.


### Basic information

The Faina dataset is released in an anonymized form (i.e., with [USER], [URL], [EMAIL], and [PHONE] placeholders) with no users' information nor original post identifier to preserve their anonymity. We include individual fallacy annotations (i.e., labels assigned by each annotator) across all instances to encourage research on human label variation, as well as information about time and topics. Labels' aggregation (if needed, e.g., via majority voting) is therefore left to the user.


### Data splits

The Faina dataset consists of two data splits: one for training/development (`data/train-dev.conll`) and one for testing (`data/test.conll`). These have been created by paying particular attention to label, time, and topic distribution across the splits to ensure reliability in the official evaluation.

- **Training/development set** [`data/train-dev.conll`]: the split for training/development purposes (80% of the posts) with gold labels. Users are free to decide how to split this set into train/dev portions as part of their design decisions.
- **Test set** [`data/test.conll`]: the split for official testing purposes (20% of the posts) without labels. To obtain official evaluation scores, the user has to submit their predictions (i.e., a file following the same format of `data/test.conll` but including the predicted labels, see [Data format](#data-format) section) through the CodaBench benchmark page (link available soon!).


### Data format

The format of the Faina dataset is based on the CoNLL data format. Each post is separated by a blank line and consists of a header with post information (i.e., id, date, topic, text), followed by each token in the text (with tab-separated information) separated by newlines. Token annotations follow the BIO scheme (i.e., `B`: begin, `I`: inside, `O`: outside) and multiple annotations for the same token and annotator are separated by a pipe (|).

Specifically, a post in the Faina dataset is represented as follows:

```
# post_id = $POST_ID
# post_date = $POST_DATE
# post_topic_keywords = $POST_TOPIC_KEYWORDS
# post_text = $POST_TEXT
$TOKEN_1      $TOKEN_1_TEXT      $TOKEN_1_LABELS_BY_ANN_A      $TOKEN_1_LABELS_BY_ANN_B
$TOKEN_2      $TOKEN_2_TEXT      $TOKEN_2_LABELS_BY_ANN_A      $TOKEN_2_LABELS_BY_ANN_B
...
$TOKEN_N      $TOKEN_N_TEXT      $TOKEN_N_LABELS_BY_ANN_A      $TOKEN_N_LABELS_BY_ANN_B
```

where:
- `$POST_ID`: the identifier of the post (*integer*);
- `$POST_DATE`: the date of the post (*YYYY-MM*);
- `$POST_TOPIC_KEYWORDS`: the set in which the keyword that led to the post selection belongs (*migration*, *climate change*, or *public health*);
- `$POST_TEXT`: the text of the post (anonymized with [USER], [URL], [EMAIL], and [PHONE] placeholders);
- `$TOKEN_i`: the index of the token within the post (*incremental integer*);
- `$TOKEN_i_TEXT`: the text of the *i*-th token within the post;
- `$TOKEN_i_LABELS_BY_ANN_j`: the fallacy label(s) assigned by a given annotator *j* for the *i*-th token within the post. Each label follows the format `$BIO`-`$LABEL`, where `$BIO` is the BIO tag and `$LABEL` is the fallacy label (e.g., "Vagueness", "Strawman"), e.g., "B-Vagueness", "I-Strawman", and "O". In the case where multiple labels for the *i*-th token are assigned by the same annotator *j*, these are separated by a pipe (|) and ordered lexicographically by `$LABEL`, e.g., "I-Strawman|B-Vagueness". In the case where no labels for the *i*-th token are assigned by the same annotator *j*, the label is "O". 

Please note that the test set does not include gold labels (i.e., it has empty `$TOKEN_i_LABELS_BY_ANN_j` columns) because it serves for official evaluation only (see [Data splits](#data-splits) section).


## Fallacy detection models

### Setup

Clone this repository on your own path:

```
git clone https://github.com/dhfbk/faina.git
```

Create an environment with your own preferred package manager. We used [python 3.9](https://www.python.org/downloads/release/python-390/) and dependencies listed in [`requirements.txt`](requirements.txt). If you use [venv](https://docs.python.org/3/library/venv.html), you can just run the following commands from the root of the project:

```
python -m venv venv                        # create the environment
source venv/bin/activate                   # activate the environment
pip install -r requirements.txt            # install the required packages
```

Get the Faina dataset (see [Faina dataset](#faina-dataset)), then put the `train-dev.conll` and `test.conll` files into the `data/` folder.


### Post-level model

Our proposed ***multi-view, multi-label*** (`MVML`) model for fine-grained, **post-level fallacy detection**. The model relies on a shared encoder and uses `D=|A|` decoders (one for each annotation view, i.e., for the labels assigned by each annotator) and outputs `D` sets of predicted labels containing all fallacy labels that exceed a threshold `τ` (with `τ=0.7`; tuned during development).

#### Data preparation

Convert the Faina dataset to a post-level format for the `MVML` model by running the following:

```
python scripts/convert-to-mvml-format.py
```

The post-level data will be stored in the `machamp/data/post-level/` folder as `train-dev.tsv` and `test.tsv`.

#### Training

For fine-tuning our proposed `MVML` model, run the following:

```
python machamp/train.py \
	--parameters_config machamp/configs/params.post-level.$ENCODER.json \
	--dataset_configs machamp/configs/data.post-level.json \
	--name mvml-model.$ENCODER \
	--device 0
```

where `$ENCODER` is either [`alberto`](https://huggingface.co/m-polignano-uniba/bert_uncased_L-12_H-768_A-12_italian_alb3rt0) or [`umberto`](https://huggingface.co/Musixmatch/umberto-commoncrawl-cased-v1). The fine-tuned model will be created at `logs/mvml-model.$ENCODER/$DATETIME/model.pt`.

#### Prediction

For predicting fallacy labels on the test set at the post-level, run the following:

```
python machamp/predict.py \
	logs/mvml-model.$ENCODER/$DATETIME/model.pt \
	machamp/data/post-level/test.tsv \
	logs/mvml-model.$ENCODER/$DATETIME/post-level.out \
	--device 0
```

You will find the predictions on `logs/mvml-model.$ENCODER/$DATETIME/post-level.out`.

#### Evaluation

The command for evaluating the performance of the model is the following:

```
python scripts/evaluate.py \
  -P logs/mvml-model.$ENCODER/$DATETIME/post-level.out \
  -G machamp/data/post-level/test-ann.tsv \
  -T post-fine
```

Here, the `test-ann.tsv` file is the same as `test.tsv` but with gold annotations. Please note that gold annotations are kept blind to avoid test set contamination in LLMs (i.e, to enable fairer comparison of future methods for the task). You can use the script above to assess the performance of your model in your own train/dev splits (by changing the `-G` parameter), while for getting the official test set scores for your method you can submit the predictions through the CodaBench benchmark page (see [Data splits](#data-splits) section).

<table>
  <tr>
    <td></td>
    <td colspan=3 align=center><i>k-fold cross validation (k=5)</i></td>
    <td colspan=3 align=center><i>official test set</i></td>
  </tr>
  <tr>
    <td></td>
    <td align=center><b>P</b></td>
    <td align=center><b>R</b></td>
    <td align=center><b>F1</b></td>
    <td align=center><b>P</b></td>
    <td align=center><b>R</b></td>
    <td align=center><b>F1</b></td>
  <tr>
    <td><b>MVML-ALB</b></td>
    <td align=center>63.0<sub>±2.0</sub></td>
    <td align=center>34.3<sub>±1.9</sub></td>
    <td align=center><b>44.3</b><sub>±1.9</sub></td>
    <td align=center>64.29</td>
    <td align=center>34.41</td>
    <td align=center><b>44.82</b></td>
  </tr>
  <tr>
    <td><b>MVML-UMB</b></td>
    <td align=center>39.0<sub>±3.7</sub></td>
    <td align=center>14.6<sub>±1.6</sub></td>
    <td align=center><b>21.3</b><sub>±2.2</sub></td>
    <td align=center>38.53</td>
    <td align=center>14.28</td>
    <td align=center><b>20.84</b></td>
  </tr>
</table>

For comparison purposes of future methods, refer to the *official test set* scores above.

### Span-level model

Our proposed ***multi-view, multi-decoder*** (`MVMD`) model for fine-grained, **span-level fallacy detection**. The model relies on a shared encoder and uses a separate decoder for each annotator view `A` and fallacy type `F` (i.e., `D = |A × F|`) and outputs `D` sets of predicted labels (i.e., either `B`, `I`, or `O` for each fallacy label and annotator view). All decoders are given equal importance in the computation of the multi-task learning loss.

#### Data preparation

Convert the Faina dataset to a span-level format (with one column per fallacy and annotator) for the `MVMD` model by running the following:

```
python scripts/convert-to-mvmd-format.py
```

The span-level data (with one column per fallacy and annotator) will be stored in the `machamp/data/span-level/` folder as `train-dev.conll` and `test.conll`.

#### Training

For fine-tuning our proposed `MVMD` model, run the following:

```
python machamp/train.py \
  --parameters_config machamp/configs/params.span-level.$ENCODER.json \
  --dataset_configs machamp/configs/data.span-level.json \
  --name mvmd-model.$ENCODER \
  --device 0
```

where `$ENCODER` is either [`alberto`](https://huggingface.co/m-polignano-uniba/bert_uncased_L-12_H-768_A-12_italian_alb3rt0) or [`umberto`](https://huggingface.co/Musixmatch/umberto-commoncrawl-cased-v1). The fine-tuned model will be created at `logs/mvmd-model.$ENCODER/$DATETIME/model.pt`.

#### Prediction

For predicting fallacy labels on the test set at the span-level, run the following:

```
python machamp/predict.py \
  logs/mvmd-model.$ENCODER/$DATETIME/model.pt \
  machamp/data/span-level/test.conll \
  logs/mvmd-model.$ENCODER/$DATETIME/span-level.out \
  --device 0
```

You will find the predictions on `logs/mvmd-model.$ENCODER/$DATETIME/span-level.out`.

#### Evaluation

The command for evaluating the performance of the model is the following:

```
python scripts/evaluate.py \
  -P logs/mvmd-model.$ENCODER/$DATETIME/span-level.out \
  -G machamp/data/span-level/test-ann.conll \
  -T span-fine \
  -M
```

Here, the `test-ann.conll` file is the same as `test.conll` but with gold annotations. Please note that gold annotations are kept blind to avoid test set contamination in LLMs (i.e, to enable fairer comparison of future methods for the task). You can use the script above to assess the performance of your model in your own train/dev splits (by changing the `-G` parameter), while for getting the official test set scores for your method you can submit the predictions through the CodaBench benchmark page (see [Data splits](#data-splits) section).

<table>
  <tr>
    <td></td>
    <td colspan=3 align=center><i>k-fold cross validation (k=5)</i></td>
    <td colspan=3 align=center><i>official test set</i></td>
  </tr>
  <tr>
    <td></td>
    <td align=center><b>P</b></td>
    <td align=center><b>R</b></td>
    <td align=center><b>F1</b></td>
    <td align=center><b>P</b></td>
    <td align=center><b>R</b></td>
    <td align=center><b>F1</b></td>
  <tr>
    <td colspan=7 align=center><i>Strict mode</i></td>
  </tr>
  <tr>
    <td><b>MVMD-ALB</b></td>
      <td align=center>47.6<sub>±1.9</sub></td>
    <td align=center>25.6<sub>±1.6</sub></td>
    <td align=center><b>33.3</b><sub>±1.4</sub></td>
    <td align=center>48.83</td>
    <td align=center>26.87</td>
    <td align=center><b>34.66</b></td>
  </tr>
  <tr>
    <td><b>MVMD-UMB</b></td>
    <td align=center>57.5<sub>±5.9</sub></td>
    <td align=center>3.9<sub>±0.7</sub></td>
    <td align=center><b>7.3</b><sub>±1.3</sub></td>
    <td align=center>60.94</td>
    <td align=center>3.05</td>
    <td align=center><b>5.80</b></td>
  </tr>
  <tr>
    <td colspan=7 align=center><i>Soft mode</i></td>
  </tr>
  <tr>
    <td><b>MVMD-ALB</b></td>
    <td align=center>52.2<sub>±2.0</sub></td>
    <td align=center>28.7<sub>±1.7</sub></td>
    <td align=center><b>37.0</b><sub>±1.5</sub></td>
    <td align=center>52.98</td>
    <td align=center>29.48</td>
    <td align=center><b>37.89</b></td>
  </tr>
  <tr>
    <td><b>MVMD-UMB</b></td>
    <td align=center>66.3<sub>±5.5</sub></td>
    <td align=center>4.8<sub>±0.7</sub></td>
    <td align=center><b>8.9</b><sub>±1.3</sub></td>
    <td align=center>65.97</td>
    <td align=center>3.28</td>
    <td align=center><b>6.25</b></td>
  </tr>
</table>

For comparison purposes of future methods, refer to the *official test set* scores above.

## Further information

If you need further information, do not hesitate to get in touch with us by writing an email to the first/corresponding author.


## Citation

If you use or build on top of this work, please cite our paper as follows:

```
@inproceedings{ramponi-etal-2025-fine,
  title = "Fine-grained Fallacy Detection with Human Label Variation",
  author = "Ramponi, Alan  and
    Daffara, Agnese  and
    Tonelli, Sara",
  editor = "Chiruzzo, Luis  and
    Ritter, Alan  and
    Wang, Lu",
  booktitle = "Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)",
  month = apr,
  year = "2025",
  address = "Albuquerque, New Mexico",
  publisher = "Association for Computational Linguistics",
  url = "https://aclanthology.org/2025.naacl-long.34/",
  doi = "10.18653/v1/2025.naacl-long.34",
  pages = "762--784",
  ISBN = "979-8-89176-189-6"
}
```
