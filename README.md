# harmful-langchain

harmful-langchain

Install python 3.11

```bash
# download
cd /tmp/
wget https://www.python.org/ftp/python/3.11.2/Python-3.11.2.tgz
tar -xzvf Python-3.11.2.tgz
cd Python-3.11.2/

# install build tools
sudo apt update
sudo apt install -y build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev

# configure, make and make install. Python3.11 will be in /usr/local/bin/python3.11
./configure --enable-optimizations
make -j `nproc`
sudo make altinstall

# make the default version as Python 3.11
sudo ln -s /usr/local/bin/python
sudo ln -s /usr/local/bin/python3.11 /usr/local/bin/python
```


```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install --upgrade -r requirements.txt
```




