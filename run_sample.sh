#! /bin/bash

if [ ! -f /env/bin/activate ]; then
    python3 -m venv env
fi

. env/bin/activate

pip install -r requirements.txt

python3 test_sample.py
