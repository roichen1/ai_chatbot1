#!/bin/bash

# Kill any process running on port 5000 (Change to 10000 if needed)
sudo fuser -k 5000/tcp

# Navigate to the application directory
cd /home/ec2-user/app || exit

# Grant execution permission to start.sh
chmod +x start.sh

# Create and activate a virtual environment if it does not exist
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Start the application in the background and redirect logs
nohup python3 app.py > app.log 2>&1 &
