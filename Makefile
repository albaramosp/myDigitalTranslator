shell:
	docker compose exec mydigitaltranslator bash

llama_cli:
	docker compose exec mydigitaltranslator uv run llama_cli.py

main:
	docker compose exec mydigitaltranslator uv run main.py