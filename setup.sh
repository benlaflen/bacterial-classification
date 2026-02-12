rm -rf ./sequences_by_category
rm -rf taxonomy_index.tsv
./bacterial-classification/install.sh
python ./bacterial-classification/build_tax_data.py
./bacterial-classification/build/targets/propogate-children ./sequences_by_category taxonomy_index.tsv