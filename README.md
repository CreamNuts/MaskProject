# MaskProject

--data_dir : 마스크를 쓴 사진 경로
--checkpoint : 학습시킨 Generation인데 아래 적은거 쓰면 될듯

python3 train.py -m test --checkpoint checkpoint/G_4600.pt --data_dir ../mask/2-with-mask.png

위 처럼 실행하면, 마스크 쓴 사진 경로에 파일명_result.jpg로 생성된 사진 나옴. 위 같은 경우는 2-with-mask_result.jpg로 생성됨