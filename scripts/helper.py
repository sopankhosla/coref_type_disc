import pickle
def get_data_pickle(filename):
	with open(filename, "rb") as f:
		x = pickle.load(f)
		
	return x
    
    
class ReverseLayerF(Function):

	@staticmethod
	def forward(ctx, x, alpha):
		ctx.alpha = alpha

		return x.view_as(x)

	@staticmethod
	def backward(ctx, grad_output):
		output = grad_output.neg() * ctx.alpha

		return output, None

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

######################## End Domain Adversarial Functions ###########################

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
						
				if True:
					matcher=re.match("#begin document \((.*)\); part (.*)$", line.rstrip())
					if matcher != None:
						docid=matcher.group(1)
						partID=matcher.group(2)

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

	return all_sents, all_ents, all_named_ents, all_antecedent_labels, all_bridging_labels, all_max_words, all_max_ents, all_doc_names, all_trees


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