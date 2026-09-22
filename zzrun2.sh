#!/usr/bin/env bash


./target/release/spm-cli  --api-client http://127.0.0.1:8082 --image test2.jpg --ask "这张图里有什么？"
#./target/release/spm-cli  --api-client http://127.0.0.1:8082 --image test2.jpg --ask "Is there fire in the picture?"
#./target/release/spm-cli  --api-client http://127.0.0.1:8082 --image test2.jpg --ask "Is there fire in the picture? Answer yes or no"
#./target/release/spm-cli  --api-client http://127.0.0.1:8082 --image test3.jpg --ask "是否有着火？只需回到是或否" 
