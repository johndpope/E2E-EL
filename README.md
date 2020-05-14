# MM-E2E-EL
Joint NER and Entity Linking

# BioBERT

Download the pretrained BioBERT-base-cased model from here https://github.com/naver/biobert-pretrained and extract it in the working directory.

# Options
```
--data_dir  Path to data directory
--model_name_or_path  Path to pretrained BioBERT
--output_dir  Path to the directory where trained model is stored
--resume_path Path to a checkpoint from where training should resume
--do_train  For training the model
--use_random_candidates  Random negative candidates will be used for Training
--use_tfidf_candidates  TF-IDF candidates will be used for Training
--use_hard_negatives  Hard negative candidates will be used for Training
--do_eval  For evaluating the model
--include_positive  Include the positive candidate in the set of potential candidates during Evaluation
--use_tfidf_candidates  TF-IDF candidates will be used for Evaluation
--use_all_candidates  All entities will be used as candidates during Evaluation
--overwrite_output_dir  Overwrite the output directory
--overwrite_cache  Overwrite the cached training and test data
```

# Training
## NER
### BioBERT + Linear
```
python run_ner.py --data_dir ./data/MM_st21pv --model_type bert --model_name_or_path ./biobert_v1.1_pubmed/model.ckpt-1000000 --output_dir ./model_output --labels ./data/MM_st21pv/MedMentions_label_list.txt --config_name ./biobert_v1.1_pubmed/bert_config.json --tokenizer_name ./biobert_v1.1_pubmed/vocab.txt --do_train --num_train_epochs 10 --overwrite_output_dir
```
### BioBERT + CRF
```
python run_crf_ner.py --data_dir ./data/MM_st21pv --model_type bert --model_name_or_path ./biobert_v1.1_pubmed/model.ckpt-1000000 --output_dir ./model_output --labels ./data/MM_st21pv/MedMentions_label_list.txt --config_name ./biobert_v1.1_pubmed/bert_config.json --tokenizer_name ./biobert_v1.1_pubmed/vocab.txt --do_train --num_train_epochs 10 --overwrite_output_dir
```
## Entity Linking
```
python run_el.py --data_dir ./data/MM_st21pv_CUI --model_type bert --model_name_or_path ./biobert_v1.1_pubmed/model.ckpt-1000000 --output_dir ./model_output --labels ./data/MM_st21pv_CUI/MedMentions_label_list.txt --config_name ./biobert_v1.1_pubmed/bert_config.json --tokenizer_name ./biobert_v1.1_pubmed/vocab.txt --do_train --num_train_epochs 10 --overwrite_output_dir
```
# Evaluation
# NER
### BioBERT + Linear
```
python run_ner.py --data_dir ./data/MM_st21pv --model_type bert --model_name_or_path ./biobert_v1.1_pubmed/model.ckpt-1000000 --output_dir ./model_output --labels ./data/MM_st21pv/MedMentions_label_list.txt --config_name ./biobert_v1.1_pubmed/bert_config.json --tokenizer_name ./biobert_v1.1_pubmed/vocab.txt --do_eval --do_predict --overwrite_output_dir
```
### BioBERT + CRF
```
python run_crf_ner.py --data_dir ./data/MM_st21pv --model_type bert --model_name_or_path ./biobert_v1.1_pubmed/model.ckpt-1000000 --output_dir ./model_output --labels ./data/MM_st21pv/MedMentions_label_list.txt --config_name ./biobert_v1.1_pubmed/bert_config.json --tokenizer_name ./biobert_v1.1_pubmed/vocab.txt --do_train --num_train_epochs 10 --overwrite_output_dir
```
