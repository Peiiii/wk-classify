rm -rf  dist/*.gz
rm -rf  dist/*.whl
rm -rf  build/*
rm -rf  wk-classify.egg-info/*
python3 setup.py sdist bdist_wheel
python3 -m twine upload dist/*