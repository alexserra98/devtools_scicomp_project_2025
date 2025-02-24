# Makefile for Pyclassify

# Run compile script
.PHONY: make
make:
	python -m pip install -r requirements.txt
	python src/pyclassify/_compile.py
	python -m pip install -e .

# Clean up by uninstalling the package
.PHONY: clean
clean:
	python -m pip uninstall pyclassify -y
