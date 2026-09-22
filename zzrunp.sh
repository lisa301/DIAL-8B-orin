#!/usr/bin/env bash 

for f in `find zzimages/ -name "*.jpg"`;do
  logf=${f}.log
  echo "decode file: $f"
  nohup ./target/release/spm-cli  --api-client http://127.0.0.1:8082 --image test2.jpg --ask "这张图里有什么？" > $logf 2>&1 &
done

wait
