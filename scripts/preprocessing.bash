export PYTHONPATH="${PYTHONPATH}:$(pwd)"

python core/util/preprocessor/argoverse_preprocess_v2.py --root dataset/ --dest dataset -s

# generate a small subset to test the training program
# python core/util/preprocessor/argoverse_preprocess_v2.py --root dataset/ --dest dataset -s
