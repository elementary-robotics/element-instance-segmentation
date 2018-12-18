# Fetch Mask R-CNN submodule and install it
git submodule update --init
cd maskrcnn && python3 setup.py install

# Install module
cd .. 
pip3 install -r requirements.txt
python3 setup.py install

mkdir -p models
wget -v -O models/sd_maskrcnn.h5 https://berkeley.box.com/shared/static/obj0b2o589gc1odr2jwkx4qjbep11t0o.h5
