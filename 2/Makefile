DOC_DIR=/home/eugene/nltk_data/corpora/reuters/training/
DICT_FILE=dictionary.txt
POSTINGS_FILE=postings.txt
QUERIES_FILE=queries.txt
OUTPUT_FILE=output.txt

index index.py:
	python index.py -i $(DOC_DIR) -d $(DICT_FILE) -p $(POSTINGS_FILE)

search search.py:
	python search.py -d $(DICT_FILE) -p $(POSTINGS_FILE) -q $(QUERIES_FILE) -o $(OUTPUT_FILE)
