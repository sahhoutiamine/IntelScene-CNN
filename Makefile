EXP ?= baseline
IMG ?= data/processed/test/forest/example.jpg

build:
	docker compose build

up:
	docker compose up -d jupyter

down:
	docker compose down

logs:
	docker compose logs -f jupyter

shell:
	docker compose run --rm trainer bash

prepare:
	docker compose run --rm trainer python -m src.data.prepare --force

train:
	docker compose run --rm trainer python -m src.training.train --exp $(EXP)

evaluate:
	docker compose run --rm trainer python -m src.evaluation.evaluate --model models/$(EXP).keras

predict:
	docker compose run --rm trainer python -m src.predict $(IMG)

test:
	docker compose run --rm trainer pytest -q
