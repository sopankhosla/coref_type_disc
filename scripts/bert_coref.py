import re
import os
from collections import Counter
import sys
import argparse
import sklearn

import pytorch_pretrained_bert
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel, BertConfig
from pytorch_pretrained_bert import BertTokenizer
from transformers import AutoTokenizer, AutoModel
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

import torch
from torch import nn
import torch.optim as optim
import numpy as np
import random
import calc_coref_metrics
from torch.autograd import grad

from torch.optim.lr_scheduler import ExponentialLR
from torch.autograd import Function

import spacy
nlp = spacy.load("en_core_web_md")

import nltk
from nltk.tree import Tree,ParentedTree
from nltk.tokenize import sent_tokenize, word_tokenize

import math

class ReverseLayerF(Function):

	@staticmethod
	def forward(ctx, x, alpha):
		ctx.alpha = alpha

		return x.view_as(x)

	@staticmethod
	def backward(ctx, grad_output):
		output = grad_output.neg() * ctx.alpha

		return output, None

HIDDEN_DIM=200

num_ctsy_map = ['cardinal', 'money', 'date', 'other']

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

tokenizer_model = "bert-base-cased"
bert_model = "bert-base-cased"
bert_dim=768

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def gradient_penalty(critic, h_s, h_t):
	# based on: https://github.com/caogang/wgan-gp/blob/master/gan_cifar10.py#L116
	alpha = torch.rand(h_s.size(0), 1).to(device)
	differences = h_t - h_s
	interpolates = h_s + (alpha * differences)
	interpolates = torch.stack([interpolates, h_s, h_t]).requires_grad_()

	preds = critic(interpolates)
	gradients = grad(preds, interpolates,
					 grad_outputs=torch.ones_like(preds),
					 retain_graph=True, create_graph=True)[0]
	gradient_norm = gradients.norm(2, dim=1)
	gradient_penalty = ((gradient_norm - 1)**2).mean()
	return gradient_penalty


######################## Start Tree Features ###########################

def get_lca_length(location1, location2):
	i = 0
	while i < len(location1) and i < len(location2) and location1[i] == location2[i]:
		i+=1
	return i

def get_labels_from_lca(ptree, lca_len, location):
	labels = []
	for i in range(lca_len, len(location)):
		labels.append(ptree[location[:i]].label())
	return labels

def findPath(ptree, text1, text2):
	leaf_values = ptree.leaves()
	leaf_index1 = leaf_values.index(text1)
	leaf_index2 = leaf_values.index(text2)

	location1 = ptree.leaf_treeposition(leaf_index1)
	location2 = ptree.leaf_treeposition(leaf_index2)

	#find length of least common ancestor (lca)
	lca_len = get_lca_length(location1, location2)

	#find path from the node1 to lca

	labels1 = get_labels_from_lca(ptree, lca_len, location1)
	#ignore the first element, because it will be counted in the second part of the path
	result = labels1[1:]
	#inverse, because we want to go from the node to least common ancestor
	result = result[::-1]
	
	labels2 = get_labels_from_lca(ptree, lca_len, location2)

	#add path from lca to node2
	result = result + labels2
	return result, lca_len, len(labels1), len(labels2)

def preOrderDistance(leaf1, leaf2, ptree):
	id1 = ptree.leaves().index(leaf1)
	id2 = ptree.leaves().index(leaf2)
	
	return id2 - id1

def distFromLCA(leaf1, leaf2, ptree):
	_,_,x,y = findPath(ptree, leaf1, leaf2)
	return x,y

def sizeCommonSubTree(leaf1, leaf2, ptree, log = False):
	if log:
		print(ptree, leaf1, leaf2)
		import pdb; pdb.set_trace()
	lca_len = findPath(ptree, leaf1, leaf2)[1]
	leaf_index1 = ptree.leaves().index(leaf1)
	
	location1 = ptree.leaf_treeposition(leaf_index1)
	
	lca_tree = ptree[location1[:lca_len]]

	if isinstance(lca_tree, int): 
		lca_tree_leaves = [lca_tree]
	else:
		lca_tree_leaves = lca_tree.leaves()
	
	return lca_len, lca_tree_leaves

def depthTree(ptree):
	nodes = ptree.treepositions(order='preorder')
	return max([len(node) for node in nodes])

def convert_num_leaves_to_sents(ptree, sents):
	leaves = ptree.leaves()

	sents_joined = [" ".join(sent[1:-1]) for sent in sents]

	data = " ".join(sents_joined)

	prev_sent = -1
	for l, leaf in enumerate(leaves):

		location = ptree.leaf_treeposition(l)

		if l < len(leaves)-1 and int(leaves[l+1]) > len(data):
			leaves[l+1] = -1
			# print(leaves)
			continue

		if leaves[l] == -1:
			continue


		if l == len(leaves) - 1:

			compare = data[min(int(leaf), len(data) - 1):].strip()

			ptree[location] = compare
			flag = 0
			for i, sent in enumerate(sents_joined):
				filt_compare = sent_tokenize(compare)[0]
				if filt_compare[int(len(filt_compare)/5): int(4*len(filt_compare)/5)] in sent or sent in filt_compare:
					ptree[location] = i
					if i >= prev_sent:
						prev_sent = i
						flag = 1
						break
			if flag == 0:
				ptree[location] = prev_sent
			
		else:
			new_ind = int(leaves[l+1])
			try:
				for i in range(min(int(leaves[l+1]), len(data)-1), int(leaves[l]), -1):
					if data[i] in [".", "!", "?"]:
						new_ind = i
						break
			except:
				import pdb; pdb.set_trace()
					
			leaves[l+1] = new_ind + 1
			
			if int(leaves[l+1]) > int(leaves[l]) + 1:

				compare = data[int(leaves[l]):int(leaves[l+1])].strip()

				ptree[location] = compare

				flag = 0
				for i, sent in enumerate(sents_joined):
					filt_compare = sent_tokenize(compare)[0]
					if filt_compare[int(len(filt_compare)/4): int(3*len(filt_compare)/4)] in sent or sent in filt_compare:
						ptree[location] = i
						if i >= prev_sent:
							prev_sent = i					
							flag = 1
							break
				if flag == 0:
					ptree[location] = prev_sent
					
			else:
				ptree[location] = prev_sent
			
				

	return ptree

######################### End Tree Features #########################

class domainCriticWSGRL(nn.Module):
	def __init__(self):
		super(domainCriticWSGRL, self).__init__()
		
		self.hidden_dim = HIDDEN_DIM
		
		self.wsgrl_FCN1 = nn.Linear(self.hidden_dim * 2, 50)
		self.wsgrl_FCN2 = nn.Linear(50, 1)
		
		self.drop_layer_020 = nn.Dropout(p=0.2)
		self.tanh = nn.Tanh()
		
	def forward(self, hidden_state):
		
		score = self.wsgrl_FCN2(self.tanh(self.drop_layer_020(self.wsgrl_FCN1(hidden_state))))
	
		return score
		
		

class LSTMTagger(BertPreTrainedModel):

	def __init__(self, config, freeze_bert=False):
		super(LSTMTagger, self).__init__(config)

		hidden_dim=HIDDEN_DIM
		self.hidden_dim=hidden_dim

		self.tokenizer = BertTokenizer.from_pretrained(tokenizer_model, do_lower_case=False, do_basic_tokenize=False)
# 		self.tokenizer = AutoTokenizer.from_pretrained("SpanBERT/spanbert-base-cased")
		self.bert = BertModel.from_pretrained(bert_model)
# 		self.bert = AutoModel.from_pretrained("SpanBERT/spanbert-base-cased")

		self.freeze_bert = freeze_bert

		if freeze_bert:
			self.bert.eval()
			for param in self.bert.parameters():
				param.requires_grad = False

		self.distance_embeddings = nn.Embedding(11, 20)
		self.nested_embeddings = nn.Embedding(2, 20)
		self.gender_embeddings = nn.Embedding(3, 20)
		self.width_embeddings = nn.Embedding(12, 20)
		self.quote_embeddings = nn.Embedding(3, 20)
# 		self.same_type_embeddings =  nn.Embedding(2, 20)
		
		mention_rep_size = 3 * 2 * hidden_dim + 20 + 20
		
		if with_type_self:
			if not common_lbl:
				self.type_embeddings = nn.Embedding(len(lbl), 20)
			else:
				self.type_embeddings = nn.Embedding(len(lbl_map_common), 20)
			mention_rep_size += 20
			
		if with_num_ctsy:
			self.men_num_ctsy_embeddings = nn.Embedding(4, 20)
			mention_rep_size += 20
			
		if with_mention_cat:
			self.mention_cat_embeddings = nn.Embedding(3, 20)
			mention_rep_size += 20
			
		mention_pair_size = mention_rep_size*3 + 20 + 20 + 20 + 20
		
		
		
		if not without_sent_distance:
			self.sent_distance_embeddings = nn.Embedding(11, 20)
		else:
			mention_pair_size -= 20
		
		if with_mention_cat:
			mention_pair_size += 20
			
		if with_overlap:
			self.mention_overlap_embeddings = nn.Embedding(2, 20)
			mention_pair_size += 20
		
		if with_type_cross:
			self.same_type_embeddings =  nn.Embedding(2, 20)
			mention_pair_size += 20
			
		if with_num_ctsy_cross:
			self.cross_num_ctsy_embeddings = nn.Embedding(3, 20)
			mention_pair_size += 20

		if with_disc_tree_feats:
			self.dist_lca_embeddings = nn.Embedding(9, 20)
			self.leaves_embeddings = nn.Embedding(5, 20)
			self.coverage_embeddings = nn.Embedding(5, 20)
			mention_pair_size += 60
			
			
		print("mention_pair_size: ", mention_pair_size)
			
		self.mention_mention1 = nn.Linear(mention_pair_size, 150)
		
		if binary_dann:
			self.domain_FCN1 = nn.Linear(mention_pair_size, 150)
				
		self.unary1 = nn.Linear(mention_rep_size, 150)
		
		if unary_dann:
			self.domain_u_FCN1 = nn.Linear(mention_rep_size, 150)
			
			
# 		if with_type_cross:
# 			self.same_type_embeddings =  nn.Embedding(2, 20)
# 			if with_type_self:
# 				mention_rep_size += 20
# 				self.mention_mention1 = nn.Linear( (3 * 2 * hidden_dim + 20 + 20 + 20) * 3 + 20 + 20 + 20 + 20 + 20, 150)
# 			else:
# 				self.mention_mention1 = nn.Linear( (3 * 2 * hidden_dim + 20 + 20) * 3 + 20 + 20 + 20 + 20 + 20, 150)
# 		else: 
# 			if with_type_self:
# 				self.mention_mention1 = nn.Linear( (3 * 2 * hidden_dim + 20 + 20 + 20) * 3 + 20 + 20 + 20 + 20, 150)	
# 				if binary_dann: #dann:
# 					self.domain_FCN1 = nn.Linear((3 * 2 * hidden_dim + 20 + 20 + 20) * 3 + 20 + 20 + 20 + 20, 150)
					
# 			else:
# 				self.mention_mention1 = nn.Linear( (3 * 2 * hidden_dim + 20 + 20) * 3 + 20 + 20 + 20 + 20, 150)
				
# 				if binary_dann:
# 					self.domain_FCN1 = nn.Linear( (3 * 2 * hidden_dim + 20 + 20) * 3 + 20 + 20 + 20 + 20, 150)
					
					

		self.lstm = nn.LSTM(4*bert_dim, hidden_dim, bidirectional=True, batch_first=True)

		self.attention1 = nn.Linear(hidden_dim * 2, hidden_dim * 2)
		self.attention2 = nn.Linear(hidden_dim * 2, 1)
		
		if pool_dann:
			self.domain_p_FCN1 = nn.Linear(hidden_dim * 2, 150)
			
		self.mention_mention2 = nn.Linear(150, 150)
		self.mention_mention3 = nn.Linear(150, 1)
		
		if binary_dann:#:	
			self.domain_FCN2 = nn.Linear(150, 150)
			self.domain_FCN3 = nn.Linear(150, num_domains)
		if unary_dann:
			self.domain_u_FCN2 = nn.Linear(150, num_domains)
# 			self.domain_u_FCN3 = nn.Linear(150, num_domains)

# 		self.domain_criterion = nn.CrossEntropyLoss(reduction='sum')
		if False:
			self.same_type_embeddings =  nn.Embedding(2, 20)
			self.domain_FCN1 = nn.Linear(mention_pair_size, 150)
			self.domain_FCN2 = nn.Linear(150, 150)
			self.domain_FCN3 = nn.Linear(150, num_domains)
		
		self.unary2 = nn.Linear(150, 150)
		self.unary3 = nn.Linear(150, 1)

		self.drop_layer_020 = nn.Dropout(p=0.2)
		self.tanh = nn.Tanh()

		self.apply(self.init_bert_weights)


	def get_mention_reps(self, input_ids=None, attention_mask=None, starts=None, ends=None, index=None, widths=None, types=None, quotes=None, num_ctsy=None, mention_cat=None, matrix=None, transforms=None, doTrain=True):

		starts=starts.to(device)
		ends=ends.to(device)
		widths=widths.to(device)
		quotes=quotes.to(device)
		types=types.to(device)
		mention_cat=mention_cat.to(device)
		
		ctsy = []
		for i, ment in enumerate(num_ctsy):
			ctsy.append(ment[1])
# 			import pdb; pdb.set_trace()
			
		ctsy=torch.LongTensor(ctsy).to(device)
			

		input_ids = input_ids.to(device)
		attention_mask = attention_mask.to(device)
		transforms = transforms.to(device)

		# matrix specifies which token positions (cols) are associated with which mention spans (row)
		matrix=matrix.to(device) # num_sents x max_ents x max_words

		# index specifies the location of the mentions in each sentence (which vary due to padding)
		index=index.to(device)

		sequence_outputs, pooled_outputs = self.bert(input_ids, token_type_ids=None, attention_mask=attention_mask)
		
		all_layers = torch.cat((sequence_outputs[-1], sequence_outputs[-2], sequence_outputs[-3], sequence_outputs[-4]), 2)
		
		
		embeds=torch.matmul(transforms,all_layers)
		
		lstm_output, _ = self.lstm(embeds) # num_sents x max_words x 2 * hidden_dim

		###########
		# ATTENTION OVER MENTION
		###########

		attention_weights=self.attention2(self.tanh(self.attention1(lstm_output))) # num_sents x max_words x 1
		attention_weights=torch.exp(attention_weights)
		
# 		if not doTrain:
# 			import pdb; pdb.set_trace()
		
		try:
			attx=attention_weights.squeeze(-1).unsqueeze(1).expand_as(matrix)
		except:
			import pdb; pdb.set_trace()
		summer=attx*matrix

		val=matrix*summer # num_sents x max_ents x max_words
		
		val=val/torch.sum(1e-8+val,dim=2).unsqueeze(-1)

		attended=torch.matmul(val, lstm_output) # num_sents x max_ents x 2 * hidden_dim

		attended=attended.view(-1,2*self.hidden_dim)

		lstm_output=lstm_output.contiguous()
		position_output=lstm_output.view(-1, 2*self.hidden_dim)
		
		# starts = token position of beginning of mention in flattened token list
		start_output=torch.index_select(position_output, 0, starts)
		# ends = token position of end of mention in flattened token list
		end_output=torch.index_select(position_output, 0, ends)

		# index = index of entity in flattened list of attended mention representations
		mentions=torch.index_select(attended, 0, index)

		try:
			width_embeds=self.width_embeddings(widths)
		except:
			print(widths.shape)
			print(widths)
			exit(0)
		quote_embeds=self.quote_embeddings(quotes)
		if with_type_self:
			type_embeds=self.type_embeddings(types)
		
			try:
				assert type_embeds.shape == width_embeds.shape
			except:
				print(type_embeds.shape)
				print(width_embeds.shape)
				
		if with_num_ctsy:
			num_ctsy_embeds=self.men_num_ctsy_embeddings(ctsy)
			
		if with_mention_cat:
			mention_cat_embeds=self.mention_cat_embeddings(mention_cat)

		if with_type_self:
			span_representation=torch.cat((start_output, end_output, mentions, width_embeds, quote_embeds, type_embeds), 1)
		else:
			span_representation=torch.cat((start_output, end_output, mentions, width_embeds, quote_embeds), 1)
			
		if with_num_ctsy:
			span_representation = torch.cat((span_representation, num_ctsy_embeds), 1)
			
		if with_mention_cat:
			span_representation = torch.cat((span_representation, mention_cat_embeds), 1)
			
		if doTrain:
			return span_representation, lstm_output
		else:
			# detach tensor from computation graph at test time or memory will blow up
			return span_representation.detach(), lstm_output


	def forward(self, matrix, index, truth=None, names=None, token_positions=None, starts=None, ends=None, widths=None, types=None, input_ids=None, attention_mask=None, transforms=None, quotes=None, num_ctsy=None, br_truth = None, mention_cat = None, training = True, targetDomain = False, domain_truth=None, tree = None, doc_sents = None, disc_v_all_sents = None):

		if with_disc_tree_feats:
			if disc_v_all_sents is None:
				disc_v_all_sents = {}

		doTrain=False
		if truth is not None:
			doTrain=True

		zeroTensor=torch.FloatTensor([0]).to(device)

		all_starts=None
		all_ends=None
		all_types=None
		all_num_ctsy=None
		all_mention_cat=None
		all_input_ids=None
		all_indices=None

		span_representation=None

		all_all=[]
		for b in range(len(matrix)):

			span_reps, lstm_output=self.get_mention_reps(input_ids=input_ids[b], attention_mask=attention_mask[b], starts=starts[b], ends=ends[b], index=index[b], widths=widths[b], types=types[b], quotes=quotes[b], num_ctsy=num_ctsy[b], mention_cat=mention_cat[b], transforms=transforms[b], matrix=matrix[b], doTrain=doTrain)
			if b == 0:
				span_representation=span_reps
				all_starts=starts[b]
				all_ends=ends[b]
				all_types=types[b]
				all_num_ctsy=num_ctsy[b]
				all_mention_cat=mention_cat[b]
				all_input_ids = input_ids[b]
				all_indices = index[b]

			else:

				span_representation=torch.cat((span_representation, span_reps), 0)
	
				all_starts=torch.cat((all_starts, starts[b]), 0)
				all_ends=torch.cat((all_ends, ends[b]), 0)
				all_types=torch.cat((all_types, types[b]), 0)
				all_num_ctsy.extend(num_ctsy[b])
				try:
					all_mention_cat=torch.cat((all_mention_cat, mention_cat[b]), 0)
				except:
					import pdb; pdb.set_trace()
				all_input_ids=torch.cat((all_input_ids, input_ids[b]), 0)
				all_indices=torch.cat((all_indices, index[b]), 0)

		all_starts=all_starts.to(device)
		all_ends=all_ends.to(device)
		all_types=all_types.to(device)
		all_mention_cat=all_mention_cat.to(device)
# 		all_input_ids=all_input_ids.to(device)
		
		num_mentions,=all_starts.shape

		running_loss=0
		
		all_domain_preds = []
		
		all_domain_truths = []

		curid=-1

		curid+=1

		assignments=[]

		seen={}

		ch=0

		token_positions=np.array(token_positions)

		mention_index=np.arange(num_mentions)

		unary_scores=self.unary3(self.tanh(self.drop_layer_020(self.unary2(self.tanh(self.drop_layer_020(self.unary1(span_representation)))))))
		
		if training and unary_dann:
			if pool_dann:			
				final_rep = torch.cat([lstm_output[:, -1, :self.hidden_dim], lstm_output[:, 0, self.hidden_dim:]], dim = -1)
				reverse_feature = ReverseLayerF.apply(final_rep, dann_alpha)
				reverse_feature = self.domain_p_FCN1(reverse_feature)
			else:
				reverse_feature = ReverseLayerF.apply(span_representation, dann_alpha)
				reverse_feature = self.domain_u_FCN1(reverse_feature)
		
			domain_preds = self.domain_u_FCN2(self.tanh(self.drop_layer_020(reverse_feature)))
# 			print(domain_preds.shape)
			domain_truth_tensor = torch.tensor([domain_truth]).to(device).repeat(domain_preds.shape[0])
# 			print(domain_truth_tensor.shape)
			domain_loss = self.domain_criterion(domain_preds, domain_truth_tensor)
# 			print("Domain loss", domain_loss)

			all_domain_preds.extend(torch.argmax(domain_preds, dim = -1).cpu().tolist())
			all_domain_truths.extend(domain_truth_tensor.cpu().tolist())
			running_loss += domain_loss
			
# 			import pdb; pdb.set_trace()

		if with_disc_tree_feats:
			# print(len(doc_sents))
			if tree is None or isinstance(tree, int):
				doc_num_leaves = 1
				doc_depth = 1
				doc_coverage = sum([len(s[1:-1]) for s in doc_sents])
			else:
				doc_num_leaves = len(tree.leaves())
				doc_depth = depthTree(tree)
				doc_coverage = sum([len(doc_sents[int(leaf)][1:-1]) for leaf in tree.leaves() if int(leaf) < len(doc_sents)])
			

		for i in range(num_mentions):
			
			if bridging: # These elements do not have bridges
				if br_truth[i] >= 1000:
					continue

			if i == 0:
				# the first mention must start a new entity; this doesn't affect training (since the loss must be 0) so we can skip it.
				if truth is None:

						assignment=curid
						curid+=1
						assignments.append(assignment)
				
				continue

			MAX_PREVIOUS_MENTIONS=300

			first=0
			if truth is None:
				if len(names[i]) == 1 and names[i][0].lower() in {"he", "his", "her", "she", "him", "they", "their", "them", "it", "himself", "its", "herself", "themselves"}:
					if "onto" in trainData.lower():
						MAX_PREVIOUS_MENTIONS=20
					else:
						MAX_PREVIOUS_MENTIONS=20

				first=i-MAX_PREVIOUS_MENTIONS
				if first < 0:
					first=0

			targets=span_representation[first:i]
			cp=span_representation[i].expand_as(targets)
			
			dists=[]
			nesteds=[]

			# get distance in mentions
			distances=i-mention_index[first:i]
			dists=vec_get_distance_bucket(distances)
			dists=torch.LongTensor(dists).to(device)
			distance_embeds=self.distance_embeddings(dists)

			# get distance in sentences
			if not without_sent_distance:
				sent_distances=token_positions[i]-token_positions[first:i]
				sent_dists=vec_get_distance_bucket(sent_distances)
				sent_dists=torch.LongTensor(sent_dists).to(device)
				sent_distance_embeds=self.sent_distance_embeddings(sent_dists)
			
			
			# Added by SK: Start #################
			# check if the type is the same
			
			# pr-processing tree features

			if with_disc_tree_feats:
				# print(token_positions[i])
				if token_positions[i] not in disc_v_all_sents:
					disc_v_all_sents[token_positions[i]] = {}
				if not isinstance(tree, int) and tree is not None:
					tree_leaves = tree.leaves()
					for e in range(first, i):
						if token_positions[e] not in disc_v_all_sents[token_positions[i]]:
							if token_positions[e] in tree_leaves and token_positions[i] in tree_leaves:
								dist_lca = distFromLCA(token_positions[i], token_positions[e], tree)[0] # TODO: Check if LCA of this mention is being used
								_, subtree_all_leaves = sizeCommonSubTree(token_positions[i], token_positions[e], tree)
								subtree_leaves = len(subtree_all_leaves)
								subtree_coverage = sum([len(doc_sents[leaf][1:-1]) for leaf in subtree_all_leaves if int(leaf) < len(doc_sents)])
							else:
								dist_lca = int(doc_depth/2)
								subtree_leaves = int(doc_num_leaves/2)
								subtree_coverage = int(doc_coverage/2)

							disc_v_all_sents[token_positions[i]][token_positions[e]] = (dist_lca, subtree_leaves, subtree_coverage)
			
			# Added by SK: End #################
			
			
			# Added by SK: Start #################
			# check if the two mentions are same/in the other
			
			if with_overlap:
				
				pair_mention_cat = []
				if len(names[i]) > 0 and ' '.join(names[i]) != "\'s":

					if all_mention_cat[i] != 2:

						cur_men = " ".join(names[i]).replace("-", " ")
						cur_men = cur_men.replace("\'s", "")
						cur_men = cur_men.strip(" ")
						try:
							if cur_men[-1] in ['s', '.']:
								cur_men = cur_men[:-1]
						except:
							import pdb; pdb.set_trace()
						for e in range(first, i):
							
							if len(names[e]) > 0 and ' '.join(names[e]) != "\'s":

								before_men = " ".join(names[e]).replace("-", " ")
								before_men = before_men.replace("\'s", "")
								before_men = before_men.strip(" ")
								if before_men[-1] in ['s', '.']:
									before_men = before_men[:-1]

								intersect = set(cur_men.split(" ")).intersection(set(before_men.split(" ")))

								if not (intersect in [set(), {"the"}]): # If there is overlap
									if intersect != (set(before_men.split(" "))) and intersect != (set(before_men.split(" ")) - {'the', 'a', 'an'}):
										# If one mention is not a substring of another
										pair_mention_cat.append(1)

									else:
										pair_mention_cat.append(0)
								else: # If there is no overlap
									pair_mention_cat.append(1)
							else:
								pair_mention_cat.append(0)

						pair_mention_cat = torch.tensor(pair_mention_cat).to(device)
					else:
						pair_mention_cat = torch.tensor([0 for xx in range(first, i)]).to(device)
						
				else:					
					pair_mention_cat = torch.tensor([0 for xx in range(first, i)]).to(device)
				
				
				mention_pair_overlap_embeds = self.mention_overlap_embeddings(pair_mention_cat)
							
			if with_mention_cat:
				mention_cat_pair_embeds = self.mention_cat_embeddings(all_mention_cat[i])
				mention_cat_pair_embeds = mention_cat_pair_embeds.repeat(i-first,1)
				
			# Added by SK: End #################
			
			# Added by SK: Start #################
			# check if the type is the same
			if with_type_cross:
				if not cosine_check:
					type_check=all_types[i] - all_types[first:i]
					type_check[type_check!=0]=1
					type_check_embeds=self.same_type_embeddings(type_check)
				else:
					x = self.type_embeddings(all_types[i])
					x = x.reshape(1, x.shape[0])
					y = self.type_embeddings(all_types[first:i])
					
					type_check = torch.nn.functional.cosine_similarity(x,y)
					type_check[type_check > 0.5]=1
					type_check[type_check <= 0.5]=0
# 					import pdb; pdb.set_trace()
# 					type_check = torch.LongTensor(type_check)
					type_check = type_check.to(device)
					type_check_embeds = type_check.reshape(type_check.shape[0], 1).repeat(1,20)
# 					import pdb; pdb.set_trace()
		

			# Added by SK: End #################
		
			# Added by SK: Start #################
			# check if the numbers are the same
			if with_num_ctsy_cross:
				num_ctsy_check = []
				for e in range(first, i):
					if all_num_ctsy[i][0] != "" and all_num_ctsy[e][0] != "":
						if all_num_ctsy[i][0] == all_num_ctsy[e][0]:
							num_ctsy_check.append(1)
						else:
							num_ctsy_check.append(0)
					else:
						num_ctsy_check.append(2)
				
				num_ctsy_check_embeds = self.cross_num_ctsy_embeddings(torch.LongTensor(num_ctsy_check).to(device))
			# Added by SK: End #################

			# Added by SK: Start #################
			# discourse tree features
			if with_disc_tree_feats:
				
				relative_dist_lca = []
				relative_leaves = []
				relative_coverage = []

				if isinstance(tree, int) or tree is None:
					for e in range(first, i):
						relative_dist_lca.append(bucket_tree_depth(doc_depth))
						relative_leaves.append(bucket_tree_leaf_cov(doc_num_leaves))
						relative_coverage.append(bucket_tree_word_cov(doc_coverage))
				else:
					for e in range(first, i):
						dist_lca, subtree_leaves, subtree_coverage = disc_v_all_sents[token_positions[i]][token_positions[e]]

						relative_dist_lca.append(bucket_tree_depth(dist_lca))
						relative_leaves.append(bucket_tree_leaf_cov(subtree_leaves))
						relative_coverage.append(bucket_tree_word_cov(subtree_coverage))

				relative_dist_lca_embeds = self.dist_lca_embeddings(torch.LongTensor(relative_dist_lca).to(device))
				relative_leaves_embeds = self.leaves_embeddings(torch.LongTensor(relative_leaves).to(device))
				relative_coverage_embeds = self.coverage_embeddings(torch.LongTensor(relative_coverage).to(device))

			# Added by SK: End #################
				

			

			# is the current mention nested within a previous one?
			res1=(all_starts[first:i] <= all_starts[i]) 
			res2=(all_ends[i] <= all_ends[first:i])

			nesteds=(res1*res2).long()
			nesteds_embeds=self.nested_embeddings(nesteds)

			res1=(all_starts[i] <= all_starts[first:i]) 
			res2=(all_ends[first:i] <= all_ends[i])

			nesteds=(res1*res2).long()
			nesteds_embeds2=self.nested_embeddings(nesteds)

			elementwise=cp*targets
			if with_type_cross:
				if not without_sent_distance:
					concat=torch.cat((cp, targets, elementwise, distance_embeds, sent_distance_embeds, nesteds_embeds, nesteds_embeds2, type_check_embeds), 1)
				else:
					concat=torch.cat((cp, targets, elementwise, distance_embeds, nesteds_embeds, nesteds_embeds2, type_check_embeds), 1)
			else:
				if not without_sent_distance:
					concat=torch.cat((cp, targets, elementwise, distance_embeds, sent_distance_embeds, nesteds_embeds, nesteds_embeds2), 1)
				else:
					concat=torch.cat((cp, targets, elementwise, distance_embeds, nesteds_embeds, nesteds_embeds2), 1)
				
# 			import pdb; pdb.set_trace()
			if with_num_ctsy_cross:
				concat = torch.cat((concat, num_ctsy_check_embeds), 1)

			if with_disc_tree_feats:
				concat = torch.cat((concat, relative_dist_lca_embeds, relative_leaves_embeds, relative_coverage_embeds), 1)
				
			if with_mention_cat:
				concat = torch.cat((concat, mention_cat_pair_embeds), 1)
				
			if with_overlap:
				concat = torch.cat((concat, mention_pair_overlap_embeds), 1)
				
# 			import pdb; pdb.set_trace()

			try:
				preds = self.mention_mention3(self.tanh(self.drop_layer_020(self.mention_mention2(self.tanh(self.drop_layer_020(self.mention_mention1(concat)))))))
			except:
				import pdb; pdb.set_trace()

			preds=preds + unary_scores[i] + unary_scores[first:i]

			preds=preds.squeeze(-1)

			if training:
				
				if not targetDomain: # TODO: Make it more general
	
					# zero is the score for the dummy antecedent/new entity
					preds=torch.cat((preds, zeroTensor))

					golds_sum=0.
					preds_sum=torch.logsumexp(preds, 0)

					tt = None

					if not bridging:
						tt = truth[i]
					else:
						tt = br_truth[i]

					if len(tt) == 1 and tt[-1] not in seen:
						golds_sum=0.
						seen[truth[i][-1]]=1
					else:
						golds=torch.index_select(preds, 0, torch.LongTensor(tt).to(device))
						golds_sum=torch.logsumexp(golds, 0)

					# want to maximize (golds_sum-preds_sum), so minimize (preds_sum-golds_sum)
					diff= (preds_sum-golds_sum)
# 					print('Classification Loss', diff)
				else:
					diff = 0
			
				################### Domain Adversarial Layer Start ###################
	
				if binary_dann:#dann:
			
					reverse_feature = ReverseLayerF.apply(concat, dann_alpha)
		
					domain_preds = self.domain_FCN3(self.tanh(self.drop_layer_020(self.domain_FCN2(self.tanh(self.drop_layer_020(self.domain_FCN1(reverse_feature)))))))
# 					print(domain_preds.shape)
					domain_truth_tensor = torch.tensor([domain_truth]).to(device).repeat(domain_preds.shape[0])
# 					print(domain_truth_tensor.shape)
					domain_loss = self.domain_criterion(domain_preds, domain_truth_tensor)
# 					print("Domain loss", domain_loss)

					all_domain_preds.extend(torch.argmax(domain_preds, dim = -1).cpu().tolist())
					all_domain_truths.extend(domain_truth_tensor.cpu().tolist())
					diff += domain_loss

				################### Domain Adversarial Layer End #####################
				
				running_loss+=diff

			else:

				assignment=None

				if i == 0:
					assignment=curid
					curid+=1

				else:

					arg_sorts=torch.argsort(preds, descending=True)
					k=0
					while k < len(arg_sorts):
						cand_idx=arg_sorts[k]
						if preds[cand_idx] > 0:
							cand_assignment=assignments[cand_idx+first]
							assignment=cand_assignment
							ch+=1
							break

						else:
							assignment=curid
							curid+=1
							break

						k+=1


				assignments.append(assignment)

		if training:
			if wsgrl:
				final_rep = torch.cat([lstm_output[:, -1, :self.hidden_dim], lstm_output[:, 0, self.hidden_dim:]], dim = -1)
				return running_loss, all_domain_preds, all_domain_truths, final_rep, disc_v_all_sents
			else:
				return running_loss, all_domain_preds, all_domain_truths, None, disc_v_all_sents
		else:
			return assignments


def get_mention_width_bucket(dist):
	if dist >= 0 and dist < 10:
		return dist + 1

	return 11

# TODO: Find the best bucketing value for the features.
def bucket_tree_depth(dist):
	if dist < 5:
		return dist + 1

	elif dist >= 5 and dist <= 7:
		return 6
	elif dist >= 8 and dist <= 10:
		return 7
	
	return 8

# TODO: Find the best bucketing value for the features.
def bucket_tree_leaf_cov(leaves):
	if leaves < 5:
		return 1
	elif leaves < 10:
		return 2
	elif leaves < 15:
		return 3
	return 4

# TODO: Find the best bucketing value for the features.
def bucket_tree_word_cov(words):
	words /= 10
	if words < 3:
		return 1
	elif words < 5:
		return 2
	elif words < 10:
		return 3
	return 4

def get_distance_bucket(dist):
	if dist < 5:
		return dist+1

	elif dist >= 5 and dist <= 7:
		return 6
	elif dist >= 8 and dist <= 15:
		return 7
	elif dist >= 16 and dist <= 31:
		return 8
	elif dist >= 32 and dist <= 63:
		return 9

	return 10

vec_get_distance_bucket=np.vectorize(get_distance_bucket)

def get_inquote(start, end, sent):

	inQuote=False
	quotes=[]

	for token in sent:
		if token == "“" or token == "\"":
			if inQuote == True:
				inQuote=False
			else:
				inQuote=True

		quotes.append(inQuote)

	for i in range(start, end+1):
		if quotes[i] == True:
			return 1

	return 0


def print_conll(name, sents, all_ents, assignments, out):

	doc_id, part_id=name

	mapper=[]
	idd=0
	for ent in all_ents:
		mapper_e=[]
		for e in ent:
			mapper_e.append(idd)
			idd+=1
		mapper.append(mapper_e)

	out.write("#begin document (%s); part %s\n" % (doc_id, part_id))
	
	for s_idx, sent in enumerate(sents):
		ents=all_ents[s_idx]
		for w_idx, word in enumerate(sent):
			if w_idx == 0 or w_idx == len(sent)-1:
				continue

			label=[]
			for idx, (start, end) in enumerate(ents):
				if start == w_idx and end == w_idx:
					eid=assignments[mapper[s_idx][idx]]
					label.append("(%s)" % eid)
				elif start == w_idx:
					eid=assignments[mapper[s_idx][idx]]
					label.append("(%s" % eid)
				elif end == w_idx:
					eid=assignments[mapper[s_idx][idx]]
					label.append("%s)" % eid)

			out.write("%s\t%s\t%s\t%s\t_\t_\t_\t_\t_\t_\t_\t_\t%s\n" % (doc_id, part_id, w_idx-1, word, '|'.join(label)))

		if len(sent) > 2:
			out.write("\n")

	out.write("#end document\n")


def test(model, test_all_docs, test_all_ents, test_all_named_ents, test_br_truth, test_all_max_words, test_all_max_ents, test_doc_names, test_all_trees, outfile, iterr, goldFile, path_to_scorer, doTest=False):

	out=open(outfile, "w", encoding="utf-8")

	# for each document
	for idx in range(len(test_all_docs)):
# 		print(idx)
		d,p=test_doc_names[idx]
# 		import pdb; pdb.set_trace()
		if only_subgenre != "":
			if only_subgenre != d.split('/')[0]:
				continue
		d=re.sub("/", "_", d)
		test_doc=test_all_docs[idx]
		test_ents=test_all_ents[idx]
		test_named_ents=test_all_named_ents[idx]

		max_words=test_all_max_words[idx]
		max_ents=test_all_max_ents[idx]

		names=[]
		for n_idx, sent in enumerate(test_ents):
			for ent in sent:
				name=test_doc[n_idx][ent[0]:ent[1]+1]
				names.append(name)

		named_index={}
		for sidx, sentence in enumerate(test_all_named_ents[idx]):
			for start, end, _ in sentence:
				named_index[(sidx, start, end)]=1

		is_named=[]

		for sidx, sentence in enumerate(test_all_ents[idx]):
			for (start, end) in sentence:
				if (sidx, start, end) in named_index:
					is_named.append(1)
				else:
					is_named.append(0)

		test_matrix, test_index, test_token_positions, test_ent_spans, test_starts, test_ends, test_widths, test_types, test_data, test_masks, test_transforms, test_quotes, test_num_ctsy, test_men_cats=get_data(model, test_doc, test_ents, test_named_ents, max_ents, max_words)
		
# 		print(test_matrix)

		try:
			assignments=model.forward(test_matrix, test_index, names=names, token_positions=test_token_positions, starts=test_starts, ends=test_ends, widths=test_widths, types=test_types, input_ids=test_data, attention_mask=test_masks, transforms=test_transforms, quotes=test_quotes, num_ctsy=test_num_ctsy, training=False, tree = test_all_trees[idx], doc_sents = test_doc, mention_cat = test_men_cats)
		except Exception as ex:
			print(ex)
			import pdb; pdb.set_trace()
			
		if not bridging:
			print_conll(test_doc_names[idx], test_doc, test_ents, assignments, out)
		else:
			print(assignments)
			import pdb; pdb.set_trace()
			
# 			for a in range(len(assignments)):
# 				if test_br_truth[idx][a]

	out.close()

	if doTest:
		print("Goldfile: %s" % goldFile)
		print("Predfile: %s" % outfile)
		
		bcub_f, avg=calc_coref_metrics.get_conll(path_to_scorer, gold=goldFile, preds=outfile)
		print("Iter %s, Average F1: %.3f, bcub F1: %s" % (iterr, avg, bcub_f))
		sys.stdout.flush()
		return avg

def get_matrix(list_of_entities, max_words, max_ents):
	matrix=np.zeros((max_ents, max_words))
	for idx, (start, end) in enumerate(list_of_entities):
		for i in range(start, end+1):
			matrix[idx,i]=1
			
# 	type_matrix=np.zeros((max_ents, max_words))
# 	for idx, (start, end, typ) in enumerate(list_of_types):
# 		for i in range(start, end+1):
# 			type_matrix[idx,i]=typ
			
	return matrix#type_matrix


def get_data(model, doc, ents, named_ents, max_ents, max_words):

	batchsize=256

	token_positions=[]
	ent_spans=[]
	persons=[]
	inquotes=[]

	batch_matrix=[]
	matrix=[]

	max_words_batch=[]
	max_ents_batch=[]

	max_w=1
	max_e=1

	sent_count=0
	for idx, sent in enumerate(doc):
		
		if len(sent) > max_w:
			max_w=len(sent)
		if len(ents[idx]) > max_e:
			max_e = len(ents[idx])

		sent_count+=1

		if sent_count == batchsize:
			max_words_batch.append(max_w)
			max_ents_batch.append(max_e)
			if max_w == 0 or max_e == 0:
				print("WTF")
			sent_count=0
			max_w=1
			max_e=1

	if sent_count > 0:
		max_words_batch.append(max_w)
		max_ents_batch.append(max_e)
		if max_w == 0 or max_e == 0:
			print("WTF")

	batch_count=0
	for idx, sent in enumerate(doc):
#         print(idx)
		mat = get_matrix(ents[idx], max_words_batch[batch_count], max_ents_batch[batch_count])
		matrix.append(mat)
		if len(matrix) == batchsize:
			batch_matrix.append(torch.FloatTensor(matrix))
			matrix=[]
			batch_count+=1
			assert batch_matrix[-1].shape[-1] != 0

	if len(matrix) > 0:
		batch_matrix.append(torch.FloatTensor(matrix))


	batch_index=[]
	batch_quotes=[]

	batch_ent_spans=[]

	index=[]
	abs_pos=0
	sent_count=0

	b=0
	for idx, sent in enumerate(ents):

		for i in range(len(sent)):
			index.append(sent_count*max_ents_batch[b] + i)
			s,e=sent[i]
			token_positions.append(idx)
			ent_spans.append(e-s)
			phrase=' '.join(doc[idx][s:e+1])

			inquotes.append(get_inquote(s, e, doc[idx]))


		abs_pos+=len(doc[idx])

		sent_count+=1

		if sent_count == batchsize:
			batch_index.append(torch.LongTensor(index))
			batch_quotes.append(torch.LongTensor(inquotes))
			batch_ent_spans.append(ent_spans)

			index=[]
			inquotes=[]
			ent_spans=[]
			sent_count=0
			b+=1

	if sent_count > 0:
		batch_index.append(torch.LongTensor(index))
		batch_quotes.append(torch.LongTensor(inquotes))
		batch_ent_spans.append(ent_spans)

	all_masks=[]
	all_transforms=[]
	all_data=[]

	batch_masks=[]
	batch_transforms=[]
	batch_data=[]

	# get ids and pad sentence
	for sent in doc:
		tok_ids=[]
		input_mask=[]
		transform=[]

		all_toks=[]
		n=0
		for idx, word in enumerate(sent):
			toks=model.tokenizer.tokenize(word)
			all_toks.append(toks)
			n+=len(toks)


		cur=0
		for idx, word in enumerate(sent):

			toks=all_toks[idx]
			ind=list(np.zeros(n))
			for j in range(cur,cur+len(toks)):
				ind[j]=1./len(toks)
			cur+=len(toks)
			transform.append(ind)

			tok_id=model.tokenizer.convert_tokens_to_ids(toks)
			assert len(tok_id) == len(toks)
			tok_ids.extend(tok_id)

			input_mask.extend(np.ones(len(toks)))

			token=word.lower()

		all_masks.append(input_mask)
		all_data.append(tok_ids)
		all_transforms.append(transform)

		if len(all_masks) == batchsize:
			batch_masks.append(all_masks)
			batch_data.append(all_data)
			batch_transforms.append(all_transforms)

			all_masks=[]
			all_data=[]
			all_transforms=[]

	if len(all_masks) > 0:
		batch_masks.append(all_masks)
		batch_data.append(all_data)
		batch_transforms.append(all_transforms)


	for b in range(len(batch_data)):

		max_len = max([len(sent) for sent in batch_data[b]])

		for j in range(len(batch_data[b])):
			
			blen=len(batch_data[b][j])

			for k in range(blen, max_len):
				batch_data[b][j].append(0)
				batch_masks[b][j].append(0)
				for z in range(len(batch_transforms[b][j])):
					batch_transforms[b][j][z].append(0)

			for k in range(len(batch_transforms[b][j]), max_words_batch[b]):
				batch_transforms[b][j].append(np.zeros(max_len))

		batch_data[b]=torch.LongTensor(batch_data[b])
		batch_transforms[b]=torch.FloatTensor(batch_transforms[b])
		batch_masks[b]=torch.FloatTensor(batch_masks[b])
		
	tok_pos=0
	starts=[]
	ends=[]
	widths=[]
	types=[]
	num_ctsy=[]
	men_cats=[]

	batch_starts=[]
	batch_ends=[]
	batch_widths=[]
	batch_types=[]
	batch_men_cats=[]
	
	prps = {"he", "this", "that", "his", "her", "she", "him", "they", "their", "them", "it", "himself", "its", "herself", "themselves"}
	
	
	batch_num_ctsy=[]

	sent_count=0
	b=0
	for idx, sent in enumerate(ents):
		s_prev = -1
		e_prev = -1
		type_prev = -1
		copied = 0
		for i in range(len(sent)):

			s,e=sent[i]

			starts.append(tok_pos+s)
			ends.append(tok_pos+e)
			
			if True:
				if with_num_ctsy or with_mention_cat:
					mention_num_rep = " ".join(doc[idx][s:e+1])
					nlped = nlp(mention_num_rep)
					flag = 0
						
					if with_mention_cat:
						if len(nlped.ents) > 0:
							men_cats.append(0)
						else:
							if mention_num_rep in prps:
								men_cats.append(2)
							else:
								men_cats.append(1)
							
					
					if with_num_ctsy:		
						for ent in nlped.ents:
							# The label comes from the map and the entity is the actual string
								if not flag and ent.label_.lower() in num_ctsy_map and ent.text.lower() == mention_num_rep.lower():
									num_ctsy.append((ent.text, num_ctsy_map.index(ent.label_.lower())))
									flag = 1
						if not flag:
							num_ctsy.append(("", len(num_ctsy_map) - 1))
							
				else:
					men_cats.append(0)
					num_ctsy.append(("", len(num_ctsy_map) - 1))
			
			try:
				if with_type_self or with_type_cross:
					if s == s_prev and e == e_prev:
# 						print("Interesting!")
						copied += 1
						try:
							types.append(int(lbl[type_prev]))
						except:
							types.append(int(type_prev))
					else:
						try:
							types.append(int(lbl[named_ents[idx][i - copied][-1]]))
						except:
							types.append(int(named_ents[idx][i - copied][-1]))
				else:
					types.append(0)
					
			except:
				import pdb; pdb.set_trace()
			widths.append(get_mention_width_bucket(e-s))
			
			type_prev = types[-1]
			
			s_prev = s
			e_prev = e

		sent_count+=1
		tok_pos+=max_words_batch[b]

		if sent_count == batchsize:
			batch_starts.append(torch.LongTensor(starts))
			batch_ends.append(torch.LongTensor(ends))
			batch_widths.append(torch.LongTensor(widths))
			batch_types.append(torch.LongTensor(types))
			batch_num_ctsy.append(num_ctsy)
			batch_men_cats.append(torch.LongTensor(men_cats))

			starts=[]
			ends=[]
			widths=[]
			types=[]
			num_ctsy=[]
			men_cats=[]
			tok_pos=0
			sent_count=0
			b+=1

	if sent_count > 0:
		batch_starts.append(torch.LongTensor(starts))
		batch_ends.append(torch.LongTensor(ends))
		batch_widths.append(torch.LongTensor(widths))
		try:
			batch_types.append(torch.LongTensor(types))
		except:
			import pdb; pdb.set_trace()
			
		batch_men_cats.append(torch.LongTensor(men_cats))
		batch_num_ctsy.append(num_ctsy)


	return batch_matrix, batch_index, token_positions, ent_spans, batch_starts, batch_ends, batch_widths, batch_types, batch_data, batch_masks, batch_transforms, batch_quotes, batch_num_ctsy, batch_men_cats


def get_ant_labels(all_doc_sents, all_doc_ents, all_doc_bridges, singleton_remove = False):

	max_words=0
	max_ents=0
	mention_id=0
	br_men_id=0

	big_ents={}
	big_br={}

	doc_antecedent_labels=[]
	doc_bridging_labels=[]
	big_doc_ents=[]
	
	if singleton_remove:
		singleton_removed_doc_ents = []
		for idx, sent in enumerate(all_doc_ents):
			all_sent_ents=sorted(all_doc_ents[idx], key=lambda x: (x[0], x[1]))
			for (s1, e1, eid1) in all_sent_ents:
				if eid1 not in antacs:
					antacs[eid1] = 0
				antacs[eid1] += 1 

		for idx, sent in enumerate(all_doc_ents):
			all_sent_ents=sorted(all_doc_ents[idx], key=lambda x: (x[0], x[1]))
			for i in range(10):
				for (s1, e1, eid1) in all_sent_ents:
					if antacs[eid1] < 2:        
						all_sent_ents.remove((s1, e1, eid1))
			singleton_removed_doc_ents.append(all_sent_ents)

		all_doc_ents = singleton_removed_doc_ents

	for idx, sent in enumerate(all_doc_sents):
		if len(sent) > max_words:
			max_words=len(sent)

		this_sent_ents=[]
		all_sent_ents=sorted(all_doc_ents[idx], key=lambda x: (x[0], x[1]))
		all_sent_br=sorted(all_doc_bridges[idx], key=lambda x: (x[0], x[1]))
		if len(all_sent_ents) > max_ents:
			max_ents=len(all_sent_ents)

		for (w_idx_start, w_idx_end, eid) in all_sent_ents:

			this_sent_ents.append((w_idx_start, w_idx_end))

			coref={}
			if eid in big_ents:
				coref=big_ents[eid]
			else:
				coref={mention_id:1}

			vals=sorted(coref.keys())

			if eid not in big_ents:
				big_ents[eid]={}

			big_ents[eid][mention_id]=1
			mention_id+=1

			doc_antecedent_labels.append(vals)

		big_doc_ents.append(this_sent_ents)
		
		if bridging:
			for (w_idx_start, w_idx_end, eid) in all_sent_br:

				brid={}
				if eid in big_br:
					brid=big_br[eid]
				else:
					brid={br_men_id:1}

				vals=sorted(brid.keys())

				if eid not in big_br:
					big_br[eid]={}

				big_br[eid][br_men_id]=1
				br_men_id+=1

				doc_bridging_labels.append(vals)
			
		else:
			doc_bridging_labels = None


	return doc_antecedent_labels, doc_bridging_labels, big_doc_ents, max_words, max_ents


def read_conll(filename, model=None, singleton_remove = False):

	docid=None
	partID=None

	# collection
	all_sents=[]
	all_ents=[]
	all_antecedent_labels=[]
	all_bridging_labels=[]
	all_max_words=[]
	all_max_ents=[]
	all_doc_names=[]

	all_trees=[]

	all_named_ents=[]
	all_nums=[]


	# for one doc
	all_doc_sents=[]
	all_doc_ents=[]
	all_doc_named_ents=[]
	all_doc_nums=[]

	# for one sentence
	sent=[]
	bridges=[]
	ents=[]
	sent.append("[SEP]")
	sid=0
	doc_count = 0

	named_ents=[]
	cur_tokens=0
	max_allowable_tokens=400
	cur_tid=0
	open_count=0
	open_br_count=0
	with open(filename, encoding="utf-8") as file:
		for line in file:
			if line.startswith("#begin document"):

				all_doc_ents=[]
				all_doc_sents=[]
				all_doc_bridges=[]
				
				all_doc_nums = []
				all_doc_named_ents=[]

				open_ents={}
				open_bridg={}
				open_named_ents={}

				sid=0
				docid=None
				
# 				if "P" in lbl:
# 					matcher=re.match("#begin document \((.*)\)", line.rstrip())
# 					if matcher != None:
# 						docid=matcher.group(1)
# 						partID=0			
				if True:
					matcher=re.match("#begin document \((.*)\); part (.*)$", line.rstrip())
					if matcher != None:
						docid=matcher.group(1)
						partID=matcher.group(2)

# 				print(docid)

			elif line.startswith("#end document"):

				doc_antecedent_labels, doc_bridging_labels, big_ents, max_words, max_ents=get_ant_labels(all_doc_sents, all_doc_ents, all_doc_bridges, singleton_remove)
				if True:
					if "onto" in modelFile.lower():
						directory = "ontonotes"
						fname = "onto"
					elif "rst" in modelFile.lower():
						directory = 'rst'
						fname = 'rst'
					elif "trains" in modelFile.lower():
						directory = 'trains'
						fname = 'trains'
						
					if 'test' in filename:
						file_mode = 'test'
					elif 'dev' in filename:
						file_mode = 'dev'
					else:
						file_mode = 'train'

					
					ptree = None
					ptree_new = None
					
					if with_disc_tree_feats:		
						if not only_wsj:
							tree_file = "../DiscourseAnalysis/txt_files/" + directory + "/" + fname + "." + file_mode + ".doc_" + str(doc_count) + ".rsttree"
						else:
							tree_file = "../DiscourseAnalysis/txt_files/" + directory + "/" + fname + "." + file_mode + ".doc_" + str(doc_count) + ".wsj.rstgt"
						
						with open(tree_file, "r") as f:
							tree_string = f.read().strip()

						print(docid, tree_file)

						if len(tree_string) > 1 and 'DOCTYPE' not in tree_string:
							ptree = ParentedTree.fromstring(tree_string)
							ptree_new = convert_num_leaves_to_sents(ptree, all_doc_sents)
						else:
							print("Only one constituent!")

					all_trees.append(ptree_new)

					doc_count += 1

					all_sents.append(all_doc_sents)
					all_ents.append(big_ents)
					
					all_nums.append(all_doc_nums)

					all_named_ents.append(all_doc_named_ents)

					all_antecedent_labels.append(doc_antecedent_labels)
					if bridging:
						all_bridging_labels.append(doc_bridging_labels)
					all_max_words.append(max_words+1)
					all_max_ents.append(max_ents+1)

					all_doc_names.append((docid,partID))



			else:

				parts=re.split("\t", line.rstrip())
				if len(parts) < 2 or (cur_tokens >= max_allowable_tokens and open_count == 0):
		
					sent.append("[CLS]")
					all_doc_sents.append(sent)
					ents=sorted(ents, key=lambda x: (x[0], x[1]))
					bridges=sorted(bridges, key=lambda x: (x[0], x[1]))
					
					if with_type_self or with_type_cross:
						try:
							assert len(ents) == len(named_ents)
						except:
							import pdb; pdb.set_trace()

					all_doc_ents.append(ents)
					all_doc_bridges.append(bridges)
					all_doc_named_ents.append(named_ents)

					ents=[]
					bridges=[]
					named_ents=[]
					nums=[]
					sent=[]
					sent.append("[SEP]")
					sid+=1

					cur_tokens=0

					cur_tid=0

					if len(parts) < 2:
						continue

				# +1 to account for initial [SEP]
				tid=cur_tid+1

				token=parts[3]
				coref=parts[-1].split("|")
				bridg=parts[-3].split("|")
				b_toks=model.tokenizer.tokenize(token)
				cur_tokens+=len(b_toks)
				cur_tid+=1

				for c in coref:
					if c.startswith("(") and c.endswith(")"):
						c=re.sub("\(", "", c)
						c=int(re.sub("\)", "", c))

						ents.append((tid, tid, c))
						

					elif c.startswith("("):
						c=int(re.sub("\(", "", c))

						if c not in open_ents:
							open_ents[c]=[]
						open_ents[c].append(tid)
						open_count+=1

					elif c.endswith(")"):
						c=int(re.sub("\)", "", c))

						try:
							assert c in open_ents
						except:
							print(c)

						start_tid=open_ents[c].pop()
						open_count-=1

						ents.append((start_tid, tid, c))
				
				if bridging:
					for c in bridg:
						if c.startswith("(") and c.endswith(")"):
							c=re.sub("\(", "", c)
							c=int(re.sub("\)", "", c))

							bridges.append((tid, tid, c))

						elif c.startswith("("):
							c=int(re.sub("\(", "", c))

							if c not in open_bridg:
								open_bridg[c]=[]
							open_bridg[c].append(tid)
							open_br_count+=1

						elif c.endswith(")"):
							c=int(re.sub("\)", "", c))

							try:
								assert c in open_bridg
							except:
								print(c)

							try:
								start_tid=open_bridg[c].pop()
							except:
								open_bridg
							open_br_count-=1

							bridges.append((start_tid, tid, c))
						
						
				ner=[i.strip() for i in parts[-2].split("|")]
					
				for c in ner:
					try:
						if c.startswith("(") and c.endswith(")"):
							c=re.sub("\(", "", c)
							c=re.sub("\)", "", c)
						
							if "P" in lbl:
								c = c[0]
						
							c=lbl[c]

							named_ents.append((tid, tid, c))

						elif c.startswith("("):
							c=re.sub("\(", "", c)
							
							if "P" in lbl:
								c = c[0]
							
							c=lbl[c]

							if c not in open_named_ents:
								open_named_ents[c]=[]
							open_named_ents[c].append(tid)

						elif c.endswith(")"):
							c=re.sub("\)", "", c)
							
							if "P" in lbl:
								c = c[0]
							
							c=lbl[c]

							assert c in open_named_ents

							start_tid=open_named_ents[c].pop()

							named_ents.append((start_tid, tid, c))
					except:
						pass

				sent.append(token)
			
	if not bridging:
		all_bridging_labels = None
			
# 	import pdb; pdb.set_trace()

	return all_sents, all_ents, all_named_ents, all_antecedent_labels, all_bridging_labels, all_max_words, all_max_ents, all_doc_names, all_trees

import pickle
def get_data_pickle(filename):
	with open(filename, "rb") as f:
		x = pickle.load(f)
		
	return x


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('-t','--trainData', help='Folder containing train data', required=False)
	parser.add_argument('-e','--email', help='Folder containing train data', required=False)
	parser.add_argument('-v','--valData', help='Folder containing test data', required=False)
	parser.add_argument('-tar','--targetData', help='Folder containing target domain data', required=False)
	parser.add_argument('-m','--mode', help='mode {train, predict}', required=False)
	parser.add_argument('-w','--model', help='modelFile (to write to or read from)', required=False)
	parser.add_argument('-nw','--new_model', help='New modelFile (to write to or read from)', required=False)
	parser.add_argument('-o','--outFile', help='outFile', required=False)
	parser.add_argument('-se','--seed', help='outFile', required=False, type=int)
	parser.add_argument('-s','--path_to_scorer', help='Path to coreference scorer', required=False)
	parser.add_argument('-wts','--with_type_self', help='Use type information?', action='store_true')
	parser.add_argument('-wto','--with_type_cross', help='Use type information?', action='store_true')
	parser.add_argument('-ncs','--with_num_ctsy', help='Use type information?', action='store_true')
	parser.add_argument('-nco','--with_num_ctsy_cross', help='Use type information?', action='store_true')
	parser.add_argument('-dis','--with_disc_tree_feats', help='Use discourse tree features?', action='store_true')
	parser.add_argument('-ner','--with_mention_cat', help='NER v prp v middle?', action='store_true')
	parser.add_argument('-over','--with_overlap', help='Overlap', action='store_true')
	parser.add_argument('-no_sd','--without_sent_distance', help='Overlap', action='store_true')
	parser.add_argument('-co','--common_type', help='Common Type?', action='store_true')
	parser.add_argument('-wsj','--only_wsj', help='Only WSJ', action='store_true')
	parser.add_argument('-os','--only_subgenre', help='Only Subgenre', default = "")
	parser.add_argument('-re','--already_reduced', help='Already reduced to common?', action='store_true')
	parser.add_argument('-nf','--no_freeze', help=' Don\'t Freeze bert parameters', action='store_true')
	parser.add_argument('-up','--use_pickle', help='Use pickle file for test', action='store_true')
	parser.add_argument('-c','--cosine_check', help='Cosine check', action='store_true')
	parser.add_argument('-p','--predicted_types', help='Predicted types', action='store_true')
	parser.add_argument('-lbl','--lbl_map', help=' Label map', required=False, type=int)
	parser.add_argument('-dmag','--dann_mag', help='Dann Alpha magnitude', required=False, type=float, default=1.0)
	parser.add_argument('-br','--bridging', help='Bridging', action='store_true')
	parser.add_argument('-dann','--dann', help='dann', action='store_true')
	parser.add_argument('-wsgrl','--wsgrl', help='wsgrl', action='store_true')
	parser.add_argument('-udann','--unary_dann', help='unary dann', action='store_true')
	parser.add_argument('-bdann','--binary_dann', help='unary dann', action='store_true')
	parser.add_argument('-pdann','--pool_dann', help='unary dann', action='store_true')
	parser.add_argument('-shuffle','--shuffle', help='shuffle', action='store_true')
	parser.add_argument('-lr','--lr', help=' lr', required=False, type=float, default=0.001)
	

	args = vars(parser.parse_args())

	mode=args["mode"]
	modelFile=args["model"]
	newModelFile=args["new_model"]
	valData=args["valData"]
	outfile=args["outFile"]
	path_to_scorer=args["path_to_scorer"]
	
	with_type_self = args["with_type_self"]
	with_type_cross = args["with_type_cross"]
	with_num_ctsy = args['with_num_ctsy']
	with_num_ctsy_cross = args['with_num_ctsy_cross']
	with_disc_tree_feats = args['with_disc_tree_feats']
	with_mention_cat = args['with_mention_cat']
	with_overlap = args['with_overlap']
	without_sent_distance = args['without_sent_distance']
	only_wsj = args['only_wsj']
	only_subgenre = args['only_subgenre']
	
	singleton_remove = False
	freeze_param = not args["no_freeze"]
	common_lbl = args["common_type"]
	lbl_num = args['lbl_map']
	use_pickle = args['use_pickle']
	reduced = args['already_reduced']
	cosine_check = args['cosine_check']
	predicted_types = args['predicted_types']
	bridging = args["bridging"]
	shuffle = args['shuffle']
	trainData=args["trainData"]
	targetData=args['targetData']
	lr = args['lr']
	
	wsgrl=args['wsgrl']
	dann_mag=args['dann_mag']
	dann = args["dann"]
	unary_dann = args["unary_dann"]
	binary_dann = args["binary_dann"]
	pool_dann = args['pool_dann']
	
	domain_map_onto = {
		'wb': 0,
		'pt': 1,
		'nw': 2,
		'bn': 3,
		'tc': 4,
		'bc': 5,
		'mz': 6
	}
	
	domain_map = {}
	
	if "onto" in trainData.lower():
		num_domains = 7
	else:
		num_domains = 1
	if dann:
		num_domains += 1
	
	if predicted_types:
		if reduced:
			suffix_pickle = "_com.pkl"
		else:
			suffix_pickle = ".pkl"
	else:
			suffix_pickle = "_ent_mark.pkl"
		
	
	if common_lbl:
		if reduced:
			lbl = lbl_map_common
		else:
			lbl = {k: lbl_map_common[v] for k,v in globals()['lbl_map' + str(lbl_num) + '_common'].items()}
	else:
		lbl = globals()['lbl_map' + str(lbl_num)]
		
	print(lbl)
	
	seed = args['seed']

	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	
	print(args)
	print(freeze_param)
	
	cache_dir = os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(0))

	model = LSTMTagger.from_pretrained(bert_model,
			  cache_dir=cache_dir,
			  freeze_bert=freeze_param)
	
	print(model)
	
	if torch.cuda.device_count() > 1:
		print("Let's use", torch.cuda.device_count(), "GPUs!")
		model = nn.DataParallel(model)
	
	if wsgrl:
		critic = domainCriticWSGRL()
		critic.to(device)
		critic_optim = torch.optim.Adam(critic.parameters(), lr=1e-4)

	config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
		num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

	model.to(device)

	optimizer = optim.Adam(model.parameters(), lr=lr)
	lr_scheduler=ExponentialLR(optimizer, gamma=0.999)

	if mode == "train" or mode =="tune":
		
		if mode == "tune":
			model.load_state_dict(torch.load(modelFile, map_location=device))
		
		
		if ("WORK_OF_ART" in lbl or use_pickle) and (with_type_self or with_type_cross):
			print("Getting data from pickle: ", trainData[:-6] + suffix_pickle)
			if use_pickle:
				all_docs, all_ents, all_named_ents, all_truth, _, all_max_words, all_max_ents, doc_ids=get_data_pickle(trainData[:-6] + suffix_pickle)
				print(len(all_named_ents))
				test_all_docs, test_all_ents, test_all_named_ents, test_all_truth, _, test_all_max_words, test_all_max_ents, test_doc_ids=get_data_pickle(valData[:-6] + suffix_pickle)
			elif "WORK_OF_ART" in lbl:
				all_docs, all_ents, all_named_ents, _, all_truth, _, all_max_words, all_max_ents, doc_ids=get_data_pickle(trainData[:-6] + suffix_pickle)
				test_all_docs, test_all_ents, test_all_named_ents, _, test_all_truth, test_all_max_words, test_all_max_ents, test_doc_ids=get_data_pickle(valData[:-6] + suffix_pickle)
		else:
			all_docs, all_ents, all_named_ents, all_truth, all_br_truth, all_max_words, all_max_ents, doc_ids, all_trees=read_conll(trainData, model,singleton_remove)
			
# 			import pdb; pdb.set_trace()
			
			if dann or wsgrl:
				target_docs, target_ents, target_named_ents, target_truth, target_br_truth, target_max_words, target_max_ents, target_doc_ids, target_trees=read_conll(targetData, model, singleton_remove)
			
			if shuffle:
				rand_ind = [i for i in range(len(all_docs))]
				random.shuffle(rand_ind)
				
				print('Random indices', len(rand_ind))
				
				all_docs = [all_docs[i] for i in rand_ind]
				all_ents = [all_ents[i] for i in rand_ind]
				all_trees = [all_trees[i] for i in rand_ind]
				if all_named_ents is not None:
					all_named_ents = [all_named_ents[i] for i in rand_ind]
				all_truth = [all_truth[i] for i in rand_ind]
				if all_br_truth is not None:
					all_br_truth = [all_br_truth[i] for i in rand_ind]
				all_max_words = [all_max_words[i] for i in rand_ind]
				all_max_ents = [all_max_ents[i] for i in rand_ind]
				doc_ids = [doc_ids[i] for i in rand_ind]
			
			test_all_docs, test_all_ents, test_all_named_ents, test_all_truth, test_br_truth, test_all_max_words, test_all_max_ents, test_doc_ids, test_trees=read_conll(valData, model,singleton_remove)
			
		domains = []
		
		for docs in doc_ids:
			if docs[0].split("/")[0] not in domains:
				domains.append(docs[0].split("/")[0])
		
		print("Number of domains: ", len(domains))
		print(domains)
		
		best_f1=0.
		cur_steps=0

		best_idx=0
		patience=20

		disc_v_all_docs = {}

		for i in range(100):

			model.train()
			bigloss=0.
			critic_loss=0.
			all_d_preds = []
			all_d_truths = []
			for idx in range(0, len(all_docs)):
				if only_subgenre != "":
					if only_subgenre != doc_ids[idx][0].split('/')[0]:
						continue
				if idx not in disc_v_all_docs:
					disc_v_all_docs[idx] = None
				if idx % 10 == 0:
					print(idx, "/", len(all_docs))
					sys.stdout.flush()
				max_words=all_max_words[idx]
				max_ents=all_max_ents[idx]

				matrix, index, token_positions, ent_spans, starts, ends, widths, types, input_ids, masks, transforms, quotes, num_ctsy, men_cats=get_data(model, all_docs[idx], all_ents[idx], all_named_ents[idx], max_ents, max_words)
				
				names=[]
				for n_idx, sent in enumerate(all_ents[idx]):
					for ent in sent:
						name=all_docs[idx][n_idx][ent[0]:ent[1]+1]
						names.append(name)
				
				if dann:
					p = float(idx + i * len(all_docs)) / 100 / len(all_docs)
					dann_alpha = dann_mag*(2. / (1. + np.exp(-10 * p)) - 1)
				else:
					dann_alpha = 0.0
				
# 				print("dann_alpha: ", dann_alpha)

				loss = 0.0

				if max_ents > 1:
					model.zero_grad()
					if bridging:
						br_tr = all_br_truth[idx]
					else:
						br_tr = None
						
					if "onto" in trainData.lower():
						domain_map = domain_map_onto
					else:
						domain_map[doc_ids[idx][0].split("/")[0]] = 0
						
					domain_truth = domain_map[doc_ids[idx][0].split("/")[0]]
						
					'''
					TODO:
					
					1. Try to push one batch of target and one batch of source documents before loss.backward()
						DONE
					2. Return and print values of classification and domain losses, and domain accuracy
						DONE
					3. Try changing the position of the adversary ??????????
					4. Change lambda
					5. Create an additional function that just outputs lstm vals for wsgrl
					
					'''
					
					if wsgrl:
						critic.eval()
					model.train()
						
						
					loss, doc_dom_preds, doc_dom_truths, s_lstm_output, disc_v_all_docs[idx] = model.forward(
												matrix, index, truth=all_truth[idx], 
												names=names, token_positions=token_positions, 
												starts=starts, ends=ends, widths=widths, 
												types=types, input_ids=input_ids, 
												attention_mask=masks, transforms=transforms, 
												quotes=quotes, num_ctsy=num_ctsy, br_truth = br_tr, 
												training = True, domain_truth = domain_truth, 
												tree = all_trees[idx], doc_sents = all_docs[idx],
												disc_v_all_sents = disc_v_all_docs[idx],
												mention_cat=men_cats)
					
					if wsgrl:
						source_lstm_output = s_lstm_output.clone()
					
					all_d_preds.extend(doc_dom_preds)
					all_d_truths.extend(doc_dom_truths)
					
				if dann or wsgrl: # Target domain prediction
						
					t_idx = idx % len(target_docs)
						
					max_words=target_max_words[t_idx]
					max_ents=target_max_ents[t_idx]
					
					if target_doc_ids[t_idx][0].split("/")[0] not in domain_map:
						domain_map[target_doc_ids[t_idx][0].split("/")[0]] = len(domain_map)	
					domain_truth = domain_map[target_doc_ids[t_idx][0].split("/")[0]]
						
					matrix, index, token_positions, ent_spans, starts, ends, widths, types, input_ids, masks, transforms, quotes, num_ctsy, men_cats=get_data(model, target_docs[t_idx], target_ents[t_idx], target_named_ents[t_idx], max_ents, max_words)
					
					if max_ents > 1:
						
						if wsgrl:
							critic.eval()
						model.train()
						
						target_loss, target_doc_dom_preds, target_doc_dom_truths, t_lstm_output, _=model.forward(
									matrix, index, truth=target_truth[t_idx], 
									names=names, token_positions=token_positions, starts=starts, 
									ends=ends, widths=widths, types=types, input_ids=input_ids, 
									attention_mask=masks, transforms=transforms, quotes=quotes, 
									num_ctsy=num_ctsy, br_truth = None, training = True, 
									targetDomain = True, domain_truth = domain_truth,
									tree = all_trees[t_idx], doc_sents = all_docs[idx],
									mention_cat = men_cats)
						
						if wsgrl:
							target_lstm_output = t_lstm_output.clone()
												

						all_d_preds.extend(target_doc_dom_preds)
						all_d_truths.extend(target_doc_dom_truths)

						loss += target_loss
						
						
						if wsgrl:
							wasserstein_distance = critic(source_lstm_output).mean() - critic(target_lstm_output).mean()
							loss += wasserstein_distance
							
					
				if max_ents > 1:
					if type(loss) != int:
						if wsgrl:
							loss.backward(retain_graph=True)
						else:
							loss.backward()
						optimizer.step()
						cur_steps+=1
						if cur_steps % 100 == 0:
							lr_scheduler.step()
						bigloss+=loss.item()
					else:
						if domain_truth is not None and domain_truth != 0:
							print(loss)
							print(domain_truth)
							print(idx)
							
				
						
						
				if wsgrl:
							
					critic.train()
					model.eval()
					
					critic_optim.zero_grad()
					for x in range(5): #(k_wsgrl):
						gp = gradient_penalty(critic, source_lstm_output.mean(0).reshape(1,source_lstm_output.shape[-1]), target_lstm_output.mean(0).reshape(1,target_lstm_output.shape[-1]))
								
						critic_s = critic(source_lstm_output)
						critic_t = critic(target_lstm_output)
						
						wasserstein_d = critic_s.mean() - critic_t.mean()
								
						critic_cost = -wasserstein_d + 10*gp #wsgrl_gamma*gp
								
						if x == 4:
							critic_cost.backward()
						else:
							critic_cost.backward(retain_graph=True)
						critic_optim.step()
						critic_loss += critic_cost.item()
					
				if False: #idx > 2900 and idx <= 7000:
					print(torch.cuda.memory_allocated() / 1024**3)
					torch.cuda.empty_cache()

			print(bigloss)
			
			print("Critic loss: ", critic_loss)
			
			print("Domain accuracy: ", sklearn.metrics.accuracy_score(np.array(all_d_preds), np.array(all_d_truths)))

			model.eval()
			doTest=False
			if i >= 2:
				doTest=True
			
			avg_f1=test(model, test_all_docs, test_all_ents, test_all_named_ents, test_br_truth, test_all_max_words, test_all_max_ents, test_doc_ids, test_trees, outfile, i, valData, path_to_scorer, doTest=doTest)

			if doTest:
				if avg_f1 > best_f1:
					model_to_save = model.module if hasattr(model, 'module') else model
					try:
						if mode == "tune":
							torch.save(model_to_save.state_dict(), newModelFile)
						else:
							torch.save(model_to_save.state_dict(), modelFile)
					except:
						try:
							if mode == "tune":
								torch.save(model_to_save.state_dict(), newModelFile)
							else:
								torch.save(model_to_save.state_dict(), modelFile)
						except:
							if mode == "tune":
								torch.save(model_to_save.state_dict(), newModelFile)
							else:
								torch.save(model_to_save.state_dict(), modelFile)
						
					print("Saving model ... %.3f is better than %.3f" % (avg_f1, best_f1))
					best_f1=avg_f1
					best_idx=i

				if i-best_idx > patience:
					print ("Stopping training at epoch %s" % i)
					break
					
			torch.cuda.empty_cache()
	
	elif mode == "predict":

		model.load_state_dict(torch.load(modelFile, map_location=device))
		model.eval()
		
		if ("WORK_OF_ART" in lbl or use_pickle) and (with_type_self or with_type_cross):
			print("Using pickle test file!")
			if use_pickle:
				test_all_docs, test_all_ents, test_all_named_ents, test_all_truth, test_all_max_words, test_all_max_ents, test_doc_ids=get_data_pickle(valData[:-6] + suffix_pickle)
			elif "WORK_OF_ART" in lbl:
				test_all_docs, test_all_ents, test_all_named_ents, _, test_all_truth, test_all_max_words, test_all_max_ents, test_doc_ids=get_data_pickle(valData[:-6] + suffix_pickle)
		else:
			test_all_docs, test_all_ents, test_all_named_ents, test_all_truth, test_br_truth, test_all_max_words, test_all_max_ents, test_doc_ids, test_trees=read_conll(valData, model,singleton_remove)

		test(model, test_all_docs, test_all_ents, test_all_named_ents, test_br_truth, test_all_max_words, test_all_max_ents, test_doc_ids, test_trees, outfile, 0, valData, path_to_scorer, doTest=True)
