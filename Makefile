.PHONY: setup test smoke

setup:
	uv sync

test:
	uv run pytest -q

# Synthetic chips -> site folds -> train one fold -> COCO segm evaluation.
smoke:
	uv run rts smoke
