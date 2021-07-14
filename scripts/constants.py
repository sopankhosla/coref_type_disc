import spacy

HIDDEN_DIM=200

TOKENIZER_MODEL = "bert-base-cased"
BERT_MODEL = "bert-base-cased"
BERT_DIM=768

num_ctsy_map = ['cardinal', 'money', 'date', 'other']

nlp = spacy.load("en_core_web_md")

prps = {"he", "this", "that", "his", "her", "she", "him", "they", "their", "them", "it", "himself", "its", "herself", "themselves"}

domain_map_onto = {
		'wb': 0,
		'pt': 1,
		'nw': 2,
		'bn': 3,
		'tc': 4,
		'bc': 5,
		'mz': 6
	}

lbl_map_common = {
	"PER": 0,
	"LOC": 1,
	"FAC": 2,
	"ORG": 3,
	"Other": 4
}

lbl_map1_common = {
	"PER": "PER",
	"LOC": "LOC",
	"FAC": "FAC",
	"GPE": "LOC",
	"VEH": "Other",
	"ORG": "ORG"
}

lbl_map2_common = {
	"P": "PER",
	"O": "ORG",
	"L": "LOC",
	"D": "Other"
}

lbl_map3_common = {'ORG': "ORG",
  'WORK_OF_ART': "Other",
  'LOC': "LOC",
  'CARDINAL': "Other",
  'EVENT': "Other",
  'NORP': "Other",
  'GPE': "LOC",
  'DATE': "Other",
  'PERSON': "PER",
  'FAC': "FAC",
  'QUANTITY': "Other",
  'ORDINAL': "Other",
  'TIME': "Other",
  'PRODUCT': "Other",
  'PERCENT': "Other",
  'MONEY': "Other",
  'LAW': "Other",
  'LANGUAGE': "Other",
  'UNK': "Other"}

lbl_map4_common = {
	"unmarked": "Other",
	"person": "PER",
	"concrete": "Other",
	"abstract": "Other",
	"time": "Other",
	"space": "Other",
	"plan": "Other",
	"numerical": "Other",
	"organization": "ORG",
	"unknown": "Other",
	"animate": "Other"
}

lbl_map5_common = {
	"Organization": "ORG",
	"Person": "PER",
	"Corporation": "FAC",
	"Event": "Other",
	"Place": "LOC",
	"Product": "Other",
	"Other": "Other",
	"Thing": "Other",
	"UNK": "Other",
	"nan": "Other"
}

lbl_map1 = {
	"PER": 0,
	"LOC": 1,
	"FAC": 2,
	"GPE": 3,
	"VEH": 4,
	"ORG": 5
}

lbl_map2 = {
	"P": 0,
	"O": 1,
	"L": 2,
	"D": 3
}

lbl_map3 = {'ORG': 0,
  'WORK_OF_ART': 1,
  'LOC': 2,
  'CARDINAL': 3,
  'EVENT': 4,
  'NORP': 5,
  'GPE': 6,
  'DATE': 7,
  'PERSON': 8,
  'FAC': 9,
  'QUANTITY': 10,
  'ORDINAL': 11,
  'TIME': 12,
  'PRODUCT': 13,
  'PERCENT': 14,
  'MONEY': 15,
  'LAW': 16,
  'LANGUAGE': 17,
  'UNK': 18}

lbl_map4 = {
	"unmarked": 0,
	"person": 1,
	"concrete": 2,
	"abstract": 3,
	"time": 4,
	"space": 5,
	"plan": 6,
	"numerical": 7,
	"organization": 8,
	"unknown": 9,
	"animate": 10
}

lbl_map5 = {
	"Organization": 0,
	"Person": 1,
	"Corporation": 2,
	"Event": 3,
	"Place": 4,
	"Product": 5,
	"Other": 6,
	"Thing": 7,
	"UNK": 8,
	"nan": 9
}