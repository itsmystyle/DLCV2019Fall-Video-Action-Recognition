# Download dataset from Dropbox
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1ncmqWLctmvecIXBdVng5cvbROoTWFSpE' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1ncmqWLctmvecIXBdVng5cvbROoTWFSpE" -O hw4_data.zip && rm -rf /tmp/cookies.txt

# Unzip the downloaded zip file
unzip ./hw4_data.zip -d hw4_data

# Remove the downloaded zip file
rm ./hw4_data.zip
