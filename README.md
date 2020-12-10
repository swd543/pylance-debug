# pylance-debug
A sample code repository for debugging pylance memory crashes

## Steps to reproduce
- Clone this repository
- Create virtual environment `python3 -m venv venv && venv/bin/pip install -r requirements.txt` (I am using python 3.6.8)
- In VS code, make pylance the default language server
- Set the virtual environment `venv` in VS code
- Pylance crashes within 2 minutes of starting up
