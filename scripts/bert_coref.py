import re
import os
from collections import Counter
import sys
import argparse
import sklearn

from helper import *

import constants

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

import nltk
from nltk.tree import Tree,ParentedTree
from nltk.tokenize import sent_tokenize, word_tokenize

import math
		
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class LSTMTagger(BertPreTrainedModel):

	def __init__(self, config, freeze_bert=False):
		super(LSTMTagger, self).__init__(config)

		hidden_dim=HIDDEN_DIM
		self.hidden_dim=hidden_dim

		self.tokenizer = BertTokenizer.from_pretrained(TOKENIZER_MODEL, do_lower_case=False, do_basic_tokenize=False)
		self.bert = BertModel.from_pretrained(BERT_MODEL)

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
 		# self.same_type_embeddings =  nn.Embedding(2, 20)
		
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
				
		self.unary1 = nn.Linear(mention_rep_size, 150)
		
		self.lstm = nn.LSTM(4*BERT_DIM, hidden_dim, bidirectional=True, batch_first=True)

		self.attention1 = nn.Linear(hidden_dim * 2, hidden_dim * 2)
		self.attention2 = nn.Linear(hidden_dim * 2, 1)
			
		self.mention_mention2 = nn.Linear(150, 150)
		self.mention_mention3 = nn.Linear(150, 1)
		
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
		# all_input_ids=all_input_ids.to(device)
		
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
					type_check = type_check.to(device)
					type_check_embeds = type_check.reshape(type_check.shape[0], 1).repeat(1,20)
		

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
				

			if with_num_ctsy_cross:
				concat = torch.cat((concat, num_ctsy_check_embeds), 1)

			if with_disc_tree_feats:
				concat = torch.cat((concat, relative_dist_lca_embeds, relative_leaves_embeds, relative_coverage_embeds), 1)
				
			if with_mention_cat:
				concat = torch.cat((concat, mention_cat_pair_embeds), 1)
				
			if with_overlap:
				concat = torch.cat((concat, mention_pair_overlap_embeds), 1)
				

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
				else:
					diff = 0
				
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
			return running_loss, all_domain_preds, all_domain_truths, None, disc_v_all_sents
		else:
			return assignments

vec_get_distance_bucket=np.vectorize(get_distance_bucket)

def test(model, test_all_docs, test_all_ents, test_all_named_ents, test_br_truth, test_all_max_words, test_all_max_ents, test_doc_names, test_all_trees, outfile, iterr, goldFile, path_to_scorer, doTest=False):

	out=open(outfile, "w", encoding="utf-8")

	# for each document
	for idx in range(len(test_all_docs)):
		d,p=test_doc_names[idx]
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
			
	# type_matrix=np.zeros((max_ents, max_words))
	# for idx, (start, end, typ) in enumerate(list_of_types):
	# 	for i in range(start, end+1):
	# 		type_matrix[idx,i]=typ
			
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
	parser.add_argument('-br','--bridging', help='Bridging', action='store_true')
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
	
	domain_map = {}
	
	if "onto" in trainData.lower():
		num_domains = 7
	else:
		num_domains = 1
	
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

	model = LSTMTagger.from_pretrained(BERT_MODEL,
			  cache_dir=cache_dir,
			  freeze_bert=freeze_param)
	
	print(model)
	
	if torch.cuda.device_count() > 1:
		print("Let's use", torch.cuda.device_count(), "GPUs!")
		model = nn.DataParallel(model)
	

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
					
					
					all_d_preds.extend(doc_dom_preds)
					all_d_truths.extend(doc_dom_truths)
							
					
				if max_ents > 1:
					if type(loss) != int:
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
					
				if False: #idx > 2900 and idx <= 7000:
					print(torch.cuda.memory_allocated() / 1024**3)
					torch.cuda.empty_cache()

			print(bigloss)

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
