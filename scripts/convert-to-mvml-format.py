import os

ORIG_DATA_FOLDER = "data"
MACHAMP_DATA_FOLDER = os.path.join("machamp", "data")
POST_LEVEL_DATA_FOLDER = "post-level"
ORIG_FILENAMES = ["train-dev.conll", "test.conll"] # "test-ann.conll"
POST_LEVEL_HEADER = "post_id\tpost_date\tpost_topic_keywords\tpost_text\tlabels_a1\tlabels_a2\n"
# For copying the test set with gold annotations, add "test-ann.conll" to ORIG_FILENAMES


for orig_filename in ORIG_FILENAMES:
	with open(os.path.join(ORIG_DATA_FOLDER, orig_filename), "r") as f:
		span_level_filename = orig_filename
		post_level_filename = orig_filename.split(".")[0] + ".tsv"
		if not os.path.exists(os.path.join(MACHAMP_DATA_FOLDER, POST_LEVEL_DATA_FOLDER)):
			os.makedirs(os.path.join(MACHAMP_DATA_FOLDER, POST_LEVEL_DATA_FOLDER))
		post_level_out = open(os.path.join(MACHAMP_DATA_FOLDER, POST_LEVEL_DATA_FOLDER, post_level_filename), "w")

		post_id = 0
		post_date = ""
		post_topic_keywords = ""
		post_text = ""
		post_fallacies_a1 = set()
		post_fallacies_a2 = set()

		post_level_out.write(POST_LEVEL_HEADER)

		for line in f:
			if line.startswith("# post_id = "):
				post_id = line.rstrip("\n").split(" = ")[1]
			elif line.startswith("# post_date = "):
				post_date = line.rstrip("\n").split(" = ")[1]
			elif line.startswith("# post_topic_keywords = "):
				post_topic_keywords = line.rstrip("\n").split(" = ")[1]
			elif line.startswith("# post_text = "):
				post_text = line.rstrip("\n")[14:]
			elif len(line) < 2:
				if len(list(post_fallacies_a1)) > 0:
					post_fallacies_a1_fine = sorted(list(post_fallacies_a1))
				if len(list(post_fallacies_a2)) > 0:
					post_fallacies_a2_fine = sorted(list(post_fallacies_a2))

				post_level_out.write(post_id + "\t" + post_date + "\t" + post_topic_keywords + "\t" + post_text + "\t" + "|".join(p for p in post_fallacies_a1_fine) + "\t" + "|".join(p for p in post_fallacies_a2_fine) + "\n")
				post_fallacies_a1 = set()
				post_fallacies_a2 = set()
				post_fallacies_a1_fine = []
				post_fallacies_a2_fine = []
			else:
				tok_id, tok_text, raw_a1_ann, raw_a2_ann = line.rstrip("\n").split("\t")
				a1_anns = raw_a1_ann.split("|")
				a2_anns = raw_a2_ann.split("|")

				for a1_ann in a1_anns:
					if a1_ann != "O":
						a1_fine_bio = a1_ann[:2]
						a1_fine_label = a1_ann[2:]
						post_fallacies_a1.add(a1_fine_label)

				for a2_ann in a2_anns:
					if a2_ann != "O":
						a2_fine_bio = a2_ann[:2]
						a2_fine_label = a2_ann[2:]
						post_fallacies_a2.add(a2_fine_label)

		post_level_out.close()
