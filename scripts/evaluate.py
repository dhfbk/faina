import argparse
import os
import sys

COARSE_LABELS = ["Distraction", "Insufficient-proof", "Simplification"]
FINE_LABELS = ["Ad-hominem", "Appeal-to-authority", "Appeal-to-emotion", "Causal-oversimplification", 
    "Cherry-picking", "Circular-reasoning", "Doubt", "Evading-the-burden-of-proof", "False-analogy", 
    "False-dilemma", "Flag-waving", "Hasty-generalization", "Loaded-language", "Name-calling-or-labelling", 
    "Red-herring", "Slippery-slope", "Slogan", "Strawman", "Thought-terminating-cliches", "Vagueness"]
SPAN_NEG_LABEL = "O"
LABELS_STRUCTURE = {
    "Ad-hominem": {"index": 0,
        "parents": ["Red-herring"],
        "grandparents": []
    },
    "Appeal-to-authority": {"index": 1,
        "parents": ["Red-herring"],
        "grandparents": []
    },
    "Appeal-to-emotion": {"index": 2,
        "parents": ["Red-herring"],
        "grandparents": []
    },
    "Causal-oversimplification": {"index": 3,
        "parents": ["Hasty-generalization"],
        "grandparents": []
    },
    "Cherry-picking": {"index": 4,
        "parents": ["Hasty-generalization", "Red-herring"],
        "grandparents": []
    },
    "Circular-reasoning": {"index": 5,
        "parents": ["Causal-oversimplification"],
        "grandparents": ["Hasty-generalization"]
    },
    "Doubt": {"index": 6,
        "parents": ["Ad-hominem"],
        "grandparents": ["Red-herring"]
    },
    "Evading-the-burden-of-proof": {"index": 7,
        "parents": [],
        "grandparents": []
    },
    "False-analogy": {"index": 8,
        "parents": ["Red-herring"],
        "grandparents": []
    },
    "False-dilemma": {"index": 9,
        "parents": ["Hasty-generalization"],
        "grandparents": []
    },
    "Flag-waving": {"index": 10,
        "parents": ["Appeal-to-emotion"],
        "grandparents": ["Red-herring"]
    },
    "Hasty-generalization": {"index": 11,
        "parents": [],
        "grandparents": []
    },
    "Loaded-language": {"index": 12,
        "parents": ["Appeal-to-emotion"],
        "grandparents": ["Red-herring"]
    },
    "Name-calling-or-labelling": {"index": 13,
        "parents": ["Ad-hominem"],
        "grandparents": ["Red-herring"]
    },
    "Red-herring": {"index": 14,
        "parents": [],
        "grandparents": []
    },
    "Slippery-slope": {"index": 15,
        "parents": ["Hasty-generalization"],
        "grandparents": []
    },
    "Slogan": {"index": 16,
        "parents": ["Appeal-to-emotion"],
        "grandparents": ["Red-herring"]
    },
    "Strawman": {"index": 17,
        "parents": ["Red-herring"],
        "grandparents": []
    },
    "Thought-terminating-cliches": {"index": 18,
        "parents": ["Appeal-to-emotion", "Hasty-generalization"],
        "grandparents": ["Red-herring"]
    },
    "Vagueness": {"index": 19,
        "parents": ["Evading-the-burden-of-proof", "Hasty-generalization"],
        "grandparents": []
    }
}

soft_matrix = [[0.0 for col in range(len(FINE_LABELS))] for row in range(len(FINE_LABELS))]
for label, details in LABELS_STRUCTURE.items():
    node_index = details["index"]
    soft_matrix[node_index][node_index] = 1.0 # prediction of same node
    for parent in details["parents"]:
        parent_index = LABELS_STRUCTURE[parent]["index"]
        soft_matrix[parent_index][node_index] = 0.5 # prediction of a parent


def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-P", "--pred_filepath", type=str, required=True,
        help="The path to the prediction file.")
    parser.add_argument("-G", "--gold_filepath", type=str, required=True,
        help="The path to the gold file.")
    parser.add_argument("-T", "--task", type=str, required=True, 
        choices=["span-fine", "span-coarse", "post-fine", "post-coarse"],
        help="The task type in the \"$LEVEL-$GRAIN\" format (e.g., \"span-fine\".")
    parser.add_argument("-M", "--merge_cols", action="store_true",
        help="Whether the label columns in the prediction file are split by fallacy type \
        and annotator and they need to be merged for evaluation. This is an option only for span-level tasks.")
    parser.add_argument("-O", "--one_pred", action="store_true",
        help="Whether the prediction file has only one prediction column.")
    args = parser.parse_args()

    return args


def get_task_labels(task_level, task_grain):
    if (task_level == "span") and (task_grain == "fine"):
        return FINE_LABELS + [SPAN_NEG_LABEL]
    elif (task_level == "span") and (task_grain == "coarse"):
        return COARSE_LABELS + [SPAN_NEG_LABEL]
    elif (task_level == "post") and (task_grain == "fine"):
        return FINE_LABELS
    elif (task_level == "post") and (task_grain == "coarse"):
        return COARSE_LABELS
    else:
        sys.exit(f"ERROR: Task level \"{task_level}\" or task grain \"{task_grain}\" is not supported.")


def get_post_annotations(filepath, task_labels):
    annotations = dict()

    with open(filepath, "r") as f:
        is_header_line = True
        for line in f:
            if is_header_line == True:
                is_header_line = False
            else:
                post_id, date, topic, text, a1_labels, a2_labels = line.rstrip("\n").split("\t")
                if post_id not in annotations.keys():
                    annotations[post_id] = {"a1": {}, "a2": {}}
                else:
                    sys.exit(f"ERROR: {post_id} is already in the annotations dict.")

                a1_labels = a1_labels.split("|")
                a2_labels = a2_labels.split("|")

                a1_empty_indexes = []
                for i in range(len(a1_labels)):
                    if a1_labels[i] == "":
                        a1_empty_indexes.append(i)
                        continue
                    if a1_labels[i] not in task_labels:
                        sys.exit(f"Label \"{a1_labels[i]}\" is not foreseen in our categorization for the task.")

                a2_empty_indexes = []
                for i in range(len(a2_labels)):
                    if a2_labels[i] == "":
                        a2_empty_indexes.append(i)
                        continue
                    if a2_labels[i] not in task_labels:
                        sys.exit(f"Label \"{a2_labels[i]}\" is not foreseen in our categorization for the task.")

                annotations[post_id]["a1"] = a1_labels
                annotations[post_id]["a2"] = a2_labels
    
    return annotations


def get_span_annotations(filepath, task_labels, merge_cols=False, one_pred=False):

    def merge_label_columns(list_of_labels):
        # Remove all "O"
        neg_count = list_of_labels.count(SPAN_NEG_LABEL)
        for i in range(neg_count):
            list_of_labels.remove(SPAN_NEG_LABEL)

        # Return the merged label
        if len(list_of_labels) > 0:
            return "|".join(list_of_labels)
        else:
            return SPAN_NEG_LABEL

    annotations = dict()

    with open(filepath, "r") as f:
        post_id = "0"
        ann_id = 0
        last_key_for_labels_a1 = {label: None for label in task_labels[:-1]}
        last_key_for_labels_a2 = {label: None for label in task_labels[:-1]}

        for line in f:
            if line.startswith("# post_id = "):
                post_id = line.rstrip("\n").split(" = ")[1]
                if post_id not in annotations.keys():
                    annotations[post_id] = {"a1": {}, "a2": {}}
                else:
                    sys.exit(f"ERROR: {post_id} is already in the annotations dict.")
                last_key_for_labels_a1 = {label: None for label in task_labels[:-1]}
                last_key_for_labels_a2 = {label: None for label in task_labels[:-1]}

            elif line.startswith("# post_date = "):
                pass

            elif line.startswith("# post_topic_keywords = "):
                pass

            elif line.startswith("# post_text = "):
                pass

            elif len(line) < 2:
                pass
                # finish

            else:
                line_parts = line.rstrip("\n").split("\t")
                if merge_cols:
                    tok_id = line_parts[0]
                    tok_text = line_parts[1]

                    # Check if there are N*2 labels (for both a1 and a2) and merge them
                    num_pos_labels = len(task_labels)-1
                    assert len(line_parts[2:]) == num_pos_labels*2
                    a1_labels = merge_label_columns(line_parts[2:(num_pos_labels+2)])
                    a2_labels = merge_label_columns(line_parts[(num_pos_labels+2):])
                else:
                    tok_id, tok_text, a1_labels, a2_labels = line_parts

                for label in a1_labels.split("|"):
                    if label == "O":
                        continue
                    else:
                        bio_part = label[:1]
                        label_part = label[2:]
                    
                    if label_part in task_labels: # "O" will be excluded above, and this will avoid not-foreseen labels
                        # We assume annotations are well-formed (a new annotation starts with "B", "I"-only annotations must be fixed before, if any)
                        if bio_part == "B":
                            ann_id += 1
                            ann_key = str(ann_id) + ":" + label_part
                            annotations[post_id]["a1"][ann_key] = {}
                            annotations[post_id]["a1"][ann_key]["label"] = label_part
                            annotations[post_id]["a1"][ann_key]["start"] = int(tok_id)
                            annotations[post_id]["a1"][ann_key]["end"] = int(tok_id)
                            last_key_for_labels_a1[label_part] = ann_key
                        elif bio_part == "I":
                            ann_key = last_key_for_labels_a1[label_part]
                            if not one_pred: # We assume annotations are consistent (overlaps of the same fallacy type must be fixed before, if any)
                                assert annotations[post_id]["a1"][ann_key]["end"] == int(tok_id)-1
                            annotations[post_id]["a1"][ann_key]["end"] = int(tok_id)
                    else:
                        sys.exit(f"Label \"{label}\" is not foreseen in our categorization for the task.")

                if not one_pred:
                    for label in a2_labels.split("|"):
                        if label == "O":
                            continue
                        else:
                            bio_part = label[:1]
                            label_part = label[2:]
                        
                        if label_part in task_labels: # "O" will be excluded above, and this will avoid not-foreseen labels
                            # We assume annotations are well-formed (a new annotation starts with "B", "I"-only annotations must be fixed before, if any)
                            if bio_part == "B":
                                ann_id += 1
                                ann_key = str(ann_id) + ":" + label_part
                                annotations[post_id]["a2"][ann_key] = {}
                                annotations[post_id]["a2"][ann_key]["label"] = label_part
                                annotations[post_id]["a2"][ann_key]["start"] = int(tok_id)
                                annotations[post_id]["a2"][ann_key]["end"] = int(tok_id)
                                last_key_for_labels_a2[label_part] = ann_key
                            elif bio_part == "I":
                                ann_key = last_key_for_labels_a2[label_part]
                                assert annotations[post_id]["a2"][ann_key]["end"] == int(tok_id)-1
                                annotations[post_id]["a2"][ann_key]["end"] = int(tok_id)
                        else:
                            print(f"Label \"{label}\" is not foreseen in our categorization for the task.")

    return annotations


def get_num_labels(annotations):
    num_pred_labels_a1 = 0
    num_pred_labels_a2 = 0
    
    for post_id, values in annotations.items():
        for annotator, ann_values in values.items():
            if annotator == "a1":
                num_pred_labels_a1 += len(ann_values)
            elif annotator == "a2":
                num_pred_labels_a2 += len(ann_values)

    return num_pred_labels_a1+num_pred_labels_a2, num_pred_labels_a1, num_pred_labels_a2


def get_num_spans(annotations):
    num_pred_spans_a1 = 0
    num_pred_spans_a2 = 0
    
    for post_id, values in annotations.items():
        for annotator, ann_values in values.items():
            if annotator == "a1":
                num_pred_spans_a1 += len(ann_values.keys())
            elif annotator == "a2":
                num_pred_spans_a2 += len(ann_values.keys())

    return num_pred_spans_a1+num_pred_spans_a2, num_pred_spans_a1, num_pred_spans_a2


def compute_post_scores(pred_labels, gold_labels, count_pred_labels, count_gold_labels, one_pred):
    assert list(pred_labels.keys()) == list(gold_labels.keys())
    post_ids = list(pred_labels.keys())

    tp_a1 = 0
    fp_a1 = 0
    fn_a1 = 0
    for post_id in post_ids:
        pred_labels_a1 = set(pred_labels[post_id]["a1"])
        gold_labels_a1 = set(gold_labels[post_id]["a1"])

        labels_in_common_a1 = pred_labels_a1.intersection(gold_labels_a1)
        labels_pred_only_a1 = pred_labels_a1.difference(labels_in_common_a1)
        labels_gold_only_a1 = gold_labels_a1.difference(labels_in_common_a1)
        tp_a1 += len(labels_in_common_a1)
        fp_a1 += len(labels_pred_only_a1)
        fn_a1 += len(labels_gold_only_a1)

    precision_a1 = tp_a1 / (tp_a1 + fp_a1)
    recall_a1 = tp_a1 / (tp_a1 + fn_a1)
    f1_a1 = (2 * precision_a1 * recall_a1) / (precision_a1 + recall_a1)

    if not one_pred:
        tp_a2 = 0
        fp_a2 = 0
        fn_a2 = 0
        for post_id in post_ids:
            pred_labels_a2 = set(pred_labels[post_id]["a2"])
            gold_labels_a2 = set(gold_labels[post_id]["a2"])

            labels_in_common_a2 = pred_labels_a2.intersection(gold_labels_a2)
            labels_pred_only_a2 = pred_labels_a2.difference(labels_in_common_a2)
            labels_gold_only_a2 = gold_labels_a2.difference(labels_in_common_a2)
            tp_a2 += len(labels_in_common_a2)
            fp_a2 += len(labels_pred_only_a2)
            fn_a2 += len(labels_gold_only_a2)

        precision_a2 = tp_a2 / (tp_a2 + fp_a2)
        recall_a2 = tp_a2 / (tp_a2 + fn_a2)
        f1_a2 = (2 * precision_a2 * recall_a2) / (precision_a2 + recall_a2)
    else:
        tp_a2 = 0
        fp_a2 = 0
        fn_a2 = 0
        for post_id in post_ids:
            pred_labels_a2 = set(pred_labels[post_id]["a1"])
            gold_labels_a2 = set(gold_labels[post_id]["a2"])

            labels_in_common_a2 = pred_labels_a2.intersection(gold_labels_a2)
            labels_pred_only_a2 = pred_labels_a2.difference(labels_in_common_a2)
            labels_gold_only_a2 = gold_labels_a2.difference(labels_in_common_a2)
            tp_a2 += len(labels_in_common_a2)
            fp_a2 += len(labels_pred_only_a2)
            fn_a2 += len(labels_gold_only_a2)

        precision_a2 = tp_a2 / (tp_a2 + fp_a2)
        recall_a2 = tp_a2 / (tp_a2 + fn_a2)
        f1_a2 = (2 * precision_a2 * recall_a2) / (precision_a2 + recall_a2)

    avg_precision = (precision_a1 + precision_a2) / 2
    avg_recall = (recall_a1 + recall_a2) / 2
    avg_f1 = (f1_a1 + f1_a2) / 2

    print(f"  \tprec\trec\tf1")
    print(f"a1\t{round(precision_a1*100, 2)}\t{round(recall_a1*100, 2)}\t{round(f1_a1*100, 2)}")
    print(f"a2\t{round(precision_a2*100, 2)}\t{round(recall_a2*100, 2)}\t{round(f1_a2*100, 2)}")
    print(f"avg\t{round(avg_precision*100, 2)}\t{round(avg_recall*100, 2)}\t{round(avg_f1*100, 2)}")


def compute_span_scores(pred_spans, gold_spans, count_pred_spans, count_gold_spans, one_pred, distance_function):

    def get_label_distance(pred_label, gold_label, distance_function):
        if distance_function == "strict":
            return 1 if (pred_label == gold_label) else 0
        elif distance_function == "soft":
            pred_index = LABELS_STRUCTURE[pred_label]["index"]
            gold_index = LABELS_STRUCTURE[gold_label]["index"]
            return soft_matrix[pred_index][gold_index]
        else:
            sys.exit(f"The distance function \"{distance_function}\" is not foreseen.")

    def get_span_comparison_score(pred_span, gold_span, measure):
        # based on the evaluation measure described in Da San Martino et al. (2019): https://aclanthology.org/D19-1565/ 
        # s \in S => pred_span
        # t \in T => gold_span
        # h => token length (for pred_spans in P; for gold_spans in R)
        # d => distance function

        pred_span_indices = set(list(range(pred_span["start"], pred_span["end"]+1)))
        gold_span_indices = set(list(range(gold_span["start"], gold_span["end"]+1)))

        if measure == "p":
            norm_factor = len(pred_span_indices)
        elif measure == "r":
            norm_factor = len(gold_span_indices)
        else:
            sys.exit(f"The C function works with \"p\" or \"r\" measure types, not \"{measure}\".")

        # Number of tokens in common between the pred and gold spans
        num_overlaps = len(pred_span_indices.intersection(gold_span_indices))

        # Weighting factor based on the distance between the pred and gold labels
        label_distance_weight = get_label_distance(pred_span["label"], gold_span["label"], distance_function)

        # Compute the full score for the comparison
        score = (num_overlaps / norm_factor) * label_distance_weight

        return score

    def get_span_precision(sum_prec, num_spans):
        score = (1 / num_spans) * sum_prec
        return score

    def get_span_recall(sum_rec, num_spans):
        score = (1 / num_spans) * sum_rec
        return score

    def get_span_f1(prec, rec):
        score = (2 * prec * rec) / (prec + rec)
        return score

    assert list(pred_spans.keys()) == list(gold_spans.keys())
    post_ids = list(pred_spans.keys())

    sum_c_precision_a1 = 0.0
    sum_c_recall_a1 = 0.0
    for post_id in post_ids:
        pred_spans_a1 = list(pred_spans[post_id]["a1"].values())
        gold_spans_a1 = list(gold_spans[post_id]["a1"].values())
        for pred_span_a1 in pred_spans_a1:
            for gold_span_a1 in gold_spans_a1:
                sum_c_precision_a1 += get_span_comparison_score(pred_span_a1, gold_span_a1, "p")
                sum_c_recall_a1 += get_span_comparison_score(pred_span_a1, gold_span_a1, "r")
    precision_a1 = get_span_precision(sum_c_precision_a1, count_pred_spans[1])
    recall_a1 = get_span_precision(sum_c_recall_a1, count_gold_spans[1])
    f1_a1 = get_span_f1(precision_a1, recall_a1)

    if not one_pred:
        sum_c_precision_a2 = 0.0
        sum_c_recall_a2 = 0.0
        for post_id in post_ids:
            pred_spans_a2 = list(pred_spans[post_id]["a2"].values())
            gold_spans_a2 = list(gold_spans[post_id]["a2"].values())
            for pred_span_a2 in pred_spans_a2:
                for gold_span_a2 in gold_spans_a2:
                    sum_c_precision_a2 += get_span_comparison_score(pred_span_a2, gold_span_a2, "p")
                    sum_c_recall_a2 += get_span_comparison_score(pred_span_a2, gold_span_a2, "r")
        precision_a2 = get_span_precision(sum_c_precision_a2, count_pred_spans[2])
        recall_a2 = get_span_precision(sum_c_recall_a2, count_gold_spans[2])
        f1_a2 = get_span_f1(precision_a2, recall_a2)
    else:
        sum_c_precision_a2 = 0.0
        sum_c_recall_a2 = 0.0
        for post_id in post_ids:
            pred_spans_a2 = list(pred_spans[post_id]["a1"].values())
            gold_spans_a2 = list(gold_spans[post_id]["a2"].values())
            for pred_span_a2 in pred_spans_a2:
                for gold_span_a2 in gold_spans_a2:
                    sum_c_precision_a2 += get_span_comparison_score(pred_span_a2, gold_span_a2, "p")
                    sum_c_recall_a2 += get_span_comparison_score(pred_span_a2, gold_span_a2, "r")
        precision_a2 = get_span_precision(sum_c_precision_a2, count_pred_spans[1])
        recall_a2 = get_span_precision(sum_c_recall_a2, count_gold_spans[2])
        f1_a2 = get_span_f1(precision_a2, recall_a2)

    avg_precision = (precision_a1 + precision_a2) / 2
    avg_recall = (recall_a1 + recall_a2) / 2
    avg_f1 = (f1_a1 + f1_a2) / 2

    print(f"  \tprec\trec\tf1")
    print(f"a1\t{round(precision_a1*100, 2)}\t{round(recall_a1*100, 2)}\t{round(f1_a1*100, 2)}")
    print(f"a2\t{round(precision_a2*100, 2)}\t{round(recall_a2*100, 2)}\t{round(f1_a2*100, 2)}")
    print(f"avg\t{round(avg_precision*100, 2)}\t{round(avg_recall*100, 2)}\t{round(avg_f1*100, 2)}")


def main():
    # Load command line arguments
    args = load_args()
    pred_filepath = args.pred_filepath
    gold_filepath = args.gold_filepath
    task_level, task_grain = args.task.split("-")
    merge_cols = args.merge_cols
    one_pred = args.one_pred

    # Get the labels for the specific task setting
    task_labels = get_task_labels(task_level, task_grain)

    # Print information to the console
    print("="*60)
    print(f"Evaluation for task [{task_level}, {task_grain}].")
    print(f"- Predictions:  {pred_filepath}")
    print(f"- Gold answers: {gold_filepath}")
    print("="*60)
    print(f"Labels for the task: {'; '.join(task_labels)}")
    print("="*60)

    # Read gold and pred annotations at the span level
    if task_level == "span":
        print("Getting the span annotations...")
        pred_spans = get_span_annotations(pred_filepath, task_labels, merge_cols, one_pred)
        count_pred_spans = get_num_spans(pred_spans)
        print(f"- Predictions:  {count_pred_spans[0]} (a1: {count_pred_spans[1]}; a2: {count_pred_spans[2]})")
        gold_spans = get_span_annotations(gold_filepath, task_labels)
        count_gold_spans = get_num_spans(gold_spans)
        print(f"- Gold answers: {count_gold_spans[0]} (a1: {count_gold_spans[1]}; a2: {count_gold_spans[2]})")
        print("-> Done.")
        print("="*60)

        print("Computing scores...")
        print("[strict evaluation]")
        compute_span_scores(pred_spans, gold_spans, count_pred_spans, count_gold_spans, one_pred, "strict")
        print("-"*60)
        print("[soft evaluation]")
        if task_grain == "fine":
            compute_span_scores(pred_spans, gold_spans, count_pred_spans, count_gold_spans, one_pred, "soft")
        print("-> Done.")
        print("="*60)

    # Read gold and pred annotations at the post level
    elif task_level == "post":
        print("Getting the post annotations...")
        pred_labels = get_post_annotations(pred_filepath, task_labels)
        count_pred_labels = get_num_labels(pred_labels)
        print(f"- Predictions:  {count_pred_labels[0]} (a1: {count_pred_labels[1]}; a2: {count_pred_labels[2]})")
        gold_labels = get_post_annotations(gold_filepath, task_labels)
        count_gold_labels = get_num_labels(gold_labels)
        print(f"- Gold answers: {count_gold_labels[0]} (a1: {count_gold_labels[1]}; a2: {count_gold_labels[2]})")
        print("-> Done.")
        print("="*60)

        print("Computing scores...")
        print("[strict evaluation]")
        compute_post_scores(pred_labels, gold_labels, count_pred_labels, count_gold_labels, one_pred)
        print("-"*60)

    else:
        sys.exit(f"ERROR: Task level {task_level} is not supported.")


if __name__ == "__main__":
    main()
