### This is a copy of La-MAML from https://github.com/montrealrobotics/La-MAML
### In order to ensure complete reproducability, we do not change the file and treat it as a baseline.

echo "Downloading Data..."
wget -nc http://cs231n.stanford.edu/tiny-imagenet-200.zip
echo "Unzipping Data..."
unzip tiny-imagenet-200.zip
echo "Last few steps..."
rm -r ./tiny-imagenet-200/test/*
python3 val_data_format.py
find . -name "*.txt" -delete
mv ./tiny-imagenet-200 ../LargeScale_Image/dat

