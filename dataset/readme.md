# 下載reviews數據集

- reviews指令
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1-X-dr2upJRaC-F8KeZ-xWbL5k2x1_gbo' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1-X-dr2upJRaC-F8KeZ-xWbL5k2x1_gbo" -O 'reviews.zip' && rm -rf /tmp/cookies.txt

- reviews_medium指令
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1vyiIumFcIds3a2jgvu9Ehoe6hYa3Xnbj' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1vyiIumFcIds3a2jgvu9Ehoe6hYa3Xnbj" -O 'reviews_medium.zip' && rm -rf /tmp/cookies.txt