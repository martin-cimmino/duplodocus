#!/bin/bash

# Configuration

# Function to count screen sessions
count_screens() {
    # Count lines that contain screen sessions (exclude header/footer lines)
    screen -ls 2>/dev/null | grep -c "^\s*[0-9]\+\."
}

echo "Starting screen monitor..."
echo "Monitoring screen sessions (runs CMD if <=1 screen exists)"
echo "Press Ctrl+C to stop"
echo ""

# Main loop
while true; do
    screen_count=$(count_screens)
    
    if [ "$screen_count" -gt 1 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] $screen_count screen sessions exist - no action needed"
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] $screen_count screen session(s) - executing command"
		TOKEN=`curl -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600"`; INSTANCE_ID=$(curl -H "X-aws-ec2-metadata-token: $TOKEN" -s http://169.254.169.254/latest/meta-data/instance-id); aws ec2 terminate-instances --instance-ids $INSTANCE_ID        
    fi
    
    sleep 30
done