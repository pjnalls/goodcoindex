#!/bin/bash

# Setup backend environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Setup frontend environment
cd frontend
nvm install 22
nvm use 22
npm install
cd ..


