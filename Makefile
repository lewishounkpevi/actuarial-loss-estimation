SHELL := /bin/bash

install-venv:
	python3.10 -m venv .actuarial-loss-prediction && pwd

# source .actuarial-loss-prediction/bin/activate

install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt
		
		
format:
	black src/*.py

lint:
	pylint --disable=R,C src/data_preprocessing.py src/model_training_evaluation.py


clean:
	rm -rf .actuarial-loss-prediction
	find -iname "*.pyc" -delete
	
	
all: install format lint
