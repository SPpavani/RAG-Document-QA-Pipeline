.PHONY: install sample ingest eval serve test docker

install:
	pip install -r requirements.txt

sample:
	python gen_pdfs.py

ingest:
	python ingest.py

eval:
	python evaluate.py

serve:
	uvicorn api:app --reload

test:
	pytest -q

docker:
	docker build -t rag-api .
