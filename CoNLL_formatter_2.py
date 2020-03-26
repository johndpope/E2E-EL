import re
import codecs

import tokenization

tokenizer = tokenization.BasicTokenizer(do_lower_case=False)

def custom_tokenizer(sentence):
	tokens = tokenizer.tokenize(sentence)
	tokens_spans = []
	start_token = 0
	end_token = 0

	for t in tokens:
		end_token = start_token + len(t)
		tokens_spans.append((start_token, end_token))
		remaining_sentence = sentence[end_token:]
		if remaining_sentence.startswith(' '):
			start_token = end_token + 1
		else:
			start_token = end_token
	return tokens, tokens_spans


IDregex = re.compile(r'^[0-9]+')

write_file = codecs.open('./data/dev.tsv', 'w+', encoding='utf-8')

with codecs.open('./data/dev_corpus.txt', 'r', encoding='utf-8') as f_train:
	reader = f_train.readlines()
	doc = []
	for line in reader:
		if len(line) < 3: # Indicates the end of document **ONLY '\r\n' **
			PubMedID = re.findall(IDregex, doc[0])
			title = doc[0].split('|')[-1].replace('\r', '')
			parts = doc[1].split('|')
			if len(parts) > 3:
				abstract = '|'.join(parts[2:])
				abstract = abstract.replace('\r', '')

			else:
				abstract = doc[1].split('|')[-1].replace('\r', '')

			named_entity_locator = {}
			for i in range(2, len(doc)): # From 3rd line to the last line of the document
				tokens = doc[i].split('\t')
				start = int(tokens[1])
				end = int(tokens[2])
				types = tokens[4]
				CUI = tokens[5].replace('\r', '').strip()

				# separate each token of the named entity and the start and end positions of them
				entity = tokens[3]
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

			title_sentences = [title.replace('\n', '')]
			abstract_sentences = abstract.replace('\n', '').split('. ')
			all_sentences = title_sentences + [abstract_sentences[idx] + '.' for idx in range(len(abstract_sentences) - 1)] + [abstract_sentences[-1]]

			for s_idx, sentence in enumerate(all_sentences):
				if s_idx == 0:
					tokens, tokens_spans = custom_tokenizer(sentence)
					offset = tokens_spans[-1][1]
				else:
					offset += 1
					tokens, tokens_spans = custom_tokenizer(sentence)
					# Add the offset to the relative spans of te tokens
					tokens_spans = [(begin + offset, end + offset) for (begin, end) in tokens_spans]

					offset = tokens_spans[-1][1]

				for t_idx, token in enumerate(tokens):
					token_start_pos = tokens_spans[t_idx][0]
					token_end_pos = tokens_spans[t_idx][1]
					if token_start_pos in named_entity_locator: # if the current tokens start postion matches any one of the entity tokens
						entity_token = named_entity_locator[token_start_pos]["entity"]
						entity_token_label = named_entity_locator[token_start_pos]["types"]
						entity_CUI = named_entity_locator[token_start_pos]['CUI']

						write_file.write(entity_token+'\t' + entity_token_label + '\t' + entity_CUI + '\n')
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
		else:
			doc.append(line)

