#!/bin/bash
# 한 번만 실행. 이미 컨테이너가 있으면 기존 것 삭제 후 새로 생성.
docker rm -f ninapro-dev 2>/dev/null

docker run -d \
  --name ninapro-dev \
  --gpus all \
  --shm-size=4g \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -v /mnt/data/projects:/mnt/data/projects \
  --workdir /mnt/data/projects/ninapro_db2_project/code \
  ninapro-emg \
  sleep infinity

echo "컨테이너 'ninapro-dev' 생성 완료"
