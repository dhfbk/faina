import os
import shutil

ORIG_DATA_FOLDER = "data"
MACHAMP_DATA_FOLDER = os.path.join("machamp", "data")
SPAN_LEVEL_DATA_FOLDER = "span-level"
ORIG_FILENAMES = ["train-dev.conll", "test.conll"]

fine_label_to_idx = {
	"Ad-hominem": 0,
	"Appeal-to-authority": 1,
	"Appeal-to-emotion": 2,
	"Causal-oversimplification": 3,
	"Cherry-picking": 4,
	"Circular-reasoning": 5,
	"Doubt": 6,
	"Evading-the-burden-of-proof": 7,
	"False-analogy": 8,
	"False-dilemma": 9,
	"Flag-waving": 10,
	"Hasty-generalization": 11,
	"Loaded-language": 12,
	"Name-calling-or-labelling": 13,
	"Red-herring": 14,
	"Slippery-slope": 15,
	"Slogan": 16,
	"Strawman": 17,
	"Thought-terminating-cliches": 18,
	"Vagueness": 19
}


for orig_filename in ORIG_FILENAMES:
	if not os.path.exists(os.path.join(MACHAMP_DATA_FOLDER, SPAN_LEVEL_DATA_FOLDER)):
		os.makedirs(os.path.join(MACHAMP_DATA_FOLDER, SPAN_LEVEL_DATA_FOLDER))
	span_level_out = open(os.path.join(MACHAMP_DATA_FOLDER, SPAN_LEVEL_DATA_FOLDER, orig_filename), "w")
	
	# For copying the test set with gold annotations, decomment the following:
	# shutil.copyfile(
	# 	os.path.join(ORIG_DATA_FOLDER, "test-ann.conll"), 
	# 	os.path.join(MACHAMP_DATA_FOLDER, SPAN_LEVEL_DATA_FOLDER, "test-ann.conll")
	# )

	with open(os.path.join(ORIG_DATA_FOLDER, orig_filename), "r") as f:
		for line in f:
			if line.startswith("# post_id = "):
				span_level_out.write(line)
			elif line.startswith("# post_date = "):
				span_level_out.write(line)
			elif line.startswith("# post_topic_keywords = "):
				span_level_out.write(line)
			elif line.startswith("# post_text = "):
				span_level_out.write(line)
			elif len(line) < 2:
				span_level_out.write(line)
			else:
				a1_cols = ["O"] * len(list(fine_label_to_idx.keys()))
				a2_cols = ["O"] * len(list(fine_label_to_idx.keys()))
				tok_id, tok_text, a1_anns, a2_anns = line.rstrip("\n").split("\t")
				a1_anns = a1_anns.split("|")
				a2_anns = a2_anns.split("|")

				for a1_ann in a1_anns:
					if (a1_ann == "O") or (a1_ann == ""):
						continue
					label_part = a1_ann[2:]
					label_index = fine_label_to_idx[label_part]
					a1_cols[label_index] = a1_ann
				a1_cols = "\t".join(a1_cols)

				for a2_ann in a2_anns:
					if (a2_ann == "O") or (a2_ann == ""):
						continue
					label_part = a2_ann[2:]
					label_index = fine_label_to_idx[label_part]
					a2_cols[label_index] = a2_ann
				a2_cols = "\t".join(a2_cols)

				span_level_out.write(tok_id + "\t" + tok_text + "\t" + a1_cols + "\t" + a2_cols + "\n")

	span_level_out.close()
