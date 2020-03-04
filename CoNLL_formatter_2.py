import re
import codecs
import nltk
#from nltk.tokenize import word_tokenize
#from nltk.tokenize import TreebankWordTokenizer
#treebank_tokenizer = TreebankWordTokenizer()
from nltk.tokenize import sent_tokenize
import string

import tokenization

tokenizer = tokenization.BasicTokenizer(do_lower_case=False)
#tokens = tokenizer.tokenize("   According to  the baseline clinical and laboratory parameters (leukocytosis of 30·109/l and more for B-ALL; and that of 100·109/l and more for T-ALL; phenotype В-I for B-ALL, phenotype Т-I-II-IV for T-ALL; LDH activity was more than twice the normal values; the presence of translocation t(4;11)), the high-risk group included most patients with B-ALL (n=110 (72.8%)) and T-ALL (n=76 (76%))\n")
#print("Tokens := ", tokens)

def custom_tokenizer(sentence):#
	#print("------------------------------------------------------------------------------")
	#print(sentence)
	tokens = tokenizer.tokenize(sentence)
	#print(tokens)
	tokens_spans = []
	start_token = 0
	end_token = 0

	for t in tokens:
		end_token = start_token + len(t)
		tokens_spans.append((start_token, end_token))
		remaining_sentence = sentence[end_token:]
		#print(remaining_sentence)
		if remaining_sentence.startswith(' '):
			start_token = end_token + 1
		else:
			start_token = end_token
	#print(tokens_spans)
	#print(tokens)
	#print(tokens_spans)	
	return tokens, tokens_spans


#print(custom_tokenizer("I am \u2037xvi's. I am a open-minded person:)\n"))


IDregex = re.compile(r'^[0-9]+')

write_file = codecs.open('./data/dev.tsv', 'w+', encoding='utf-8')

with codecs.open('./data/dev_corpus.txt', 'r', encoding='utf-8') as f_train:
	reader = f_train.readlines()
	doc = []
	for line in reader:
		#print(len(line))
		if len(line) < 3: # Indicates the end of document **ONLY '\r\n' **
			PubMedID = re.findall(IDregex, doc[0])
			title = doc[0].split('|')[-1].replace('\r', '')
			parts = doc[1].split('|')
			if len(parts) > 3:
				abstract = '|'.join(parts[2:])
				abstract = abstract.replace('\r', '')

			else:
				abstract = doc[1].split('|')[-1].replace('\r', '')


			#title_abstract = title + abstract
			#print(title_abstract)
			#print(len(doc))
			named_entity_locator = {}
			for i in range(2, len(doc)): # From 3rd line to the last line of the document
				tokens = doc[i].split('\t')
				start = int(tokens[1])
				end = int(tokens[2])
				types = tokens[4]
				CUI = tokens[5].replace('\r', '').strip()

				# separate each token of the named entity and the start and end positions of them
				entity = tokens[3]
				#entity_tokens = treebank_tokenizer.tokenize(entity)
				#entity_tokens_spans = list(treebank_tokenizer.span_tokenize(entity))
				entity_tokens, entity_tokens_spans = custom_tokenizer(entity)

				e_t_start = start
				for t_idx, e_t in enumerate(entity_tokens):
					# Add the relative spans to the absolute start position
					e_t_start = start + entity_tokens_spans[t_idx][0] 
					e_t_end = start + entity_tokens_spans[t_idx][1]
					if t_idx == 0: # If it is the first token then the label starts with 'B'
						named_entity_locator[e_t_start] = {"end": e_t_end, "entity": e_t, "types": 'B-'+types, "CUI": CUI}
					else: # If it is **not** the first token then the label starts with 'I'
						named_entity_locator[e_t_start] = {"end": e_t_end, "entity": e_t, "types": 'I-'+types, "CUI": CUI}

			#print(named_entity_locator)

			# Find the postion of spaces in the text
			#pos_of_spaces = [m.start() for m in re.finditer(r'\s', title_abstract)]
			#print(pos_of_spaces)
			# Find the start postion of the tokens in the text
			#pos_of_start_tokens = [0] + [n+1 for n in pos_of_spaces]
			#print(pos_of_start_tokens)
			#print(len(pos_of_start_tokens))
			#print(len(re.split(r'\s', title_abstract)))
			#title_sentences = sent_tokenize(title)
			#abstact_sentences = sent_tokenize(abstract)
			title_sentences = [title.replace('\n', '')]
			abstract_sentences = abstract.replace('\n', '').split('. ')
			all_sentences = title_sentences + [abstract_sentences[idx] + '.' for idx in range(len(abstract_sentences) - 1)] + [abstract_sentences[-1]]
			#print(all_sentences)

			#all_tokens = []
			#all_tokens_spans = []
			for s_idx, sentence in enumerate(all_sentences):
				if s_idx == 0: 
					#tokens = treebank_tokenizer.tokenize(sentence)
					tokens, tokens_spans = custom_tokenizer(sentence)
					#print(tokens)
					#print(tokens_spans)
					#all_tokens += tokens
					#print(len(all_tokens))
					#tokens_spans = list(treebank_tokenizer.span_tokenize(sentence))
					#all_tokens_spans += tokens_spans
					#print(len(all_tokens_spans))
					offset = tokens_spans[-1][1]
					#print(offset)
				else:
					offset += 1
					#print(offset)
					#tokens = treebank_tokenizer.tokenize(sentence)
					tokens, tokens_spans = custom_tokenizer(sentence)
					#print(tokens)
					#print(tokens_spans)
					#all_tokens += tokens
					#print(len(all_tokens))
					#tokens_spans = list(treebank_tokenizer.span_tokenize(sentence))
					# Add the offset to the relative spans of te tokens
					tokens_spans = [(begin + offset, end + offset) for (begin, end) in tokens_spans]

					#all_tokens_spans += tokens_spans
					offset = tokens_spans[-1][1]

				for t_idx, token in enumerate(tokens):
					token_start_pos = tokens_spans[t_idx][0]
					token_end_pos = tokens_spans[t_idx][1]
					if token_start_pos in named_entity_locator: # if the current tokens start postion matches any one of the entity tokens
						entity_token = named_entity_locator[token_start_pos]["entity"]
						entity_token_label = named_entity_locator[token_start_pos]["types"]
						entity_CUI = named_entity_locator[token_start_pos]['CUI']
						#print(entity_token+'\t'+ entity_token_label)
						write_file.write(entity_token+'\t'+ entity_token_label + '\t' + entity_CUI + '\n')
					else:
						write_file.write(token + '\tO \t None \n')

				write_file.write('\n')

			'''
			# CoNLL formatting
			title_end = len(title)
			#print(title_end)
			for t_idx, token in enumerate(all_tokens):
				token_start_pos = all_tokens_spans[t_idx][0]
				token_end_pos = all_tokens_spans[t_idx][1]
				if token_start_pos in named_entity_locator: # if the current tokens start postion matches any one of the entity tokens
					entity_token = named_entity_locator[token_start_pos]["entity"]
					entity_token_label = named_entity_locator[token_start_pos]["types"]
					#print(entity_token+'\t'+ entity_token_label)
					write_file.write(entity_token+'\t'+ entity_token_label + '\n')
				else:
					if token == '.':
						#print(token + '\tO')
						write_file.write(token + '\tO\n')
						write_file.write('\n')
					else:
						#print(token + '\tO')
						write_file.write(token + '\tO\n')

				# Identify the end of the title sentence
				if token_end_pos == title_end - 1:
					#print("\n")
					write_file.write('\n')
			'''

			doc = []
			#break
		else:
			doc.append(line)
			#print(doc)

