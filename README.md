# MM-E2E-EL
Joint NER and Entity Linking

# BioBERT

Download the pretrained BioBERT-base-cased model from here https://github.com/naver/biobert-pretrained and extract it in the working directory.

# Training
## BioBERT + Linear
```
python run_ner.py --data_dir ./data/MM_st21pv --model_type bert --model_name_or_path ./biobert_v1.1_pubmed/model.ckpt-1000000 --output_dir ./model_output --labels ./data/MM_st21pv/MedMentions_label_list.txt --config_name ./biobert_v1.1_pubmed/bert_config.json --tokenizer_name ./biobert_v1.1_pubmed/vocab.txt --do_train --num_train_epochs 10 --overwrite_output_dir
```
## BioBERT + CRF
```
python run_crf_ner.py --data_dir ./data/MM_st21pv --model_type bert --model_name_or_path ./biobert_v1.1_pubmed/model.ckpt-1000000 --output_dir ./model_output --labels ./data/MM_st21pv/MedMentions_label_list.txt --config_name ./biobert_v1.1_pubmed/bert_config.json --tokenizer_name ./biobert_v1.1_pubmed/vocab.txt --do_train --num_train_epochs 10 --overwrite_output_dir
```
## Entity Linking
```
python run_el.py --data_dir ./data/MM_st21pv_CUI --model_type bert --model_name_or_path ./biobert_v1.1_pubmed/model.ckpt-1000000 --output_dir ./model_output --labels ./data/MM_st21pv_CUI/MedMentions_label_list.txt --config_name ./biobert_v1.1_pubmed/bert_config.json --tokenizer_name ./biobert_v1.1_pubmed/vocab.txt --do_train --num_train_epochs 10 --overwrite_output_dir
```
# Evaluation
## BioBERT + Linear
```
python run_ner.py --data_dir ./data/MM_st21pv --model_type bert --model_name_or_path ./biobert_v1.1_pubmed/model.ckpt-1000000 --output_dir ./model_output --labels ./data/MM_st21pv/MedMentions_label_list.txt --config_name ./biobert_v1.1_pubmed/bert_config.json --tokenizer_name ./biobert_v1.1_pubmed/vocab.txt --do_eval --do_predict --overwrite_output_dir
```
## BioBERT + CRF
```
python run_crf_ner.py --data_dir ./data/MM_st21pv --model_type bert --model_name_or_path ./biobert_v1.1_pubmed/model.ckpt-1000000 --output_dir ./model_output --labels ./data/MM_st21pv/MedMentions_label_list.txt --config_name ./biobert_v1.1_pubmed/bert_config.json --tokenizer_name ./biobert_v1.1_pubmed/vocab.txt --do_train --num_train_epochs 10 --overwrite_output_dir
```
