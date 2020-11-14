import os
raw_data_dir = "./data/aida-yago2-dataset/raw_data"
candidates_data_dir = os.path.join(raw_data_dir, 'AIDA_candidates')

mention_candidates = {}
entities = {}

with open(os.path.join(raw_data_dir, 'entities.txt')) as f:
    for line in f:
        eid, _, entity_text = line.strip().split('\t')
        if eid not in entities:
            entities[eid] = entity_text

for dir in os.listdir(candidates_data_dir):
    for file in os.listdir(os.path.join(candidates_data_dir, dir)):
        with open(os.path.join(candidates_data_dir, dir, file), encoding='utf-8') as f:

            for line in f:
                if line.startswith("ENTITY"):
                    cols = line.strip().split('\t')
                    mention_text = cols[1].split(":")[-1]
                    if mention_text not in mention_candidates:
                        mention_candidates[mention_text] = []
                if line.startswith("CANDIDATE"):
                    cols = line.strip().split('\t')
                    candidate_id = cols[1].split(":")[-1]
                    mention_candidates[mention_text].append(candidate_id)

                    candidate_name = cols[6].split(":")[-1]
                    if candidate_id not in entities:
                        entities[candidate_id] = candidate_name

with open(os.path.join(raw_data_dir, 'entities.txt'), 'w+', encoding='utf-8') as f:
    for eid in entities:
        f.write(eid + '\tUNK\t' + entities[eid] + '\n')