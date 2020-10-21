# put all the data in the forms that are required

import json

file_names = [
    "hike"
]
file_exts = [
    ".dmrs",
    ".tags",
    ".tree",
]

directory = "../mrs-processing/data/extracted/train/"


def get_spans(tree):
    sentence = []
    for token in tree["tokens"]:
        sentence.append(token)
    # print(json.dumps(tree1, indent=4))
    # print(sentence)
    spans = []
    # create the model input
    for node in tree["nodes"]:
        fr = node["anchors"][0]["from"]
        to = node["anchors"][0]["end"]
        label = node["label"]
        # print(fr)
        # print(to)
        # print(label)
        span = sentence[fr:to + 1]
        spans.append([span, label])
    return spans


with open(directory + file_names[0] + file_exts[0], "r") as dmrs_data:
    all_data = dmrs_data.readlines()
    end_train = 270

    spans = []

    for tree in all_data[:end_train]:
        tree = json.loads(tree)
        spans.append(get_spans(tree))

    with open("temp_span_data_train.txt", "w+", encoding="cp1252") as output_file:
        for span in spans:
            for item in span:
                output_file.write(" ".join([word["lemma"] for word in item[0]]) + "|" + item[1] + "\n")

        output_file.close()

    spans = []

    for tree in all_data[end_train:]:
        tree = json.loads(tree)
        spans.append(get_spans(tree))

    with open("temp_span_data_test.txt", "w+", encoding="cp1252") as output_file:
        for span in spans:
            for item in span:
                output_file.write(" ".join([word["lemma"] for word in item[0]]) + "|" + item[1] + "\n")

        output_file.close()
