# crm-vision
This repo is for prototype a crm based on face recognition and others techniques related with computer vision.

## System requirements
ubuntu 18 - 20

python >= 3.7

## Clone repo
<pre>
git clone https://github.com/Crismarquez/crm-vision.git
cd crm-vision
</pre> 

## Virtual enviroment
<pre>
python3 -m venv .venv
source .venv/bin/activate
</pre> 

## Install dependencies
<pre>
pip install -r requirements.txt
export PYTHONPATH="${PYTHONPATH}:${PWD}"
</pre> 

## Run demo
### For register a new user
<pre>
python3 main_register.py
</pre> 

### For face recognition system
<pre>
python3 main.py
</pre> 