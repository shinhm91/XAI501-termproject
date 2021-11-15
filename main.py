from xai.data.mvtec import MVTec_AD


# load val, train data
DB_PATH = "./dataset/MVTecAD"
OUT_PATH = "./out/"

size = (86, 86)
cls = 'bottle'
dataset = MVTec_AD(DB_PATH, OUT_PATH, flatten=True, size=size)

train = dataset.read_mvtec(cls=cls, mode='train')
val = dataset.read_mvtec(cls=cls, mode='val')
test = dataset.read_mvtec(cls=cls, mode='test')

# 해당 데이터 셋은 비지도 학습을 위한 데이터 셋이기 때문에,
# train data는 정상 케이스의 영상들로만 구성되어 있습니다.
# 베이스 코드에서는 train의 key 정보와 같이 train의 라벨을 주어주지 않습니다.
# 데이터 구성에 대한 내용은 https://colab.research.google.com/drive/1pdgvoPs3KDLq6pV9oLxkXDh6waEp76HT?usp=sharing에 있습니다. 해당 부분도 보시기 바랍니다.
# 위의 URL 주소는 Dataset 설명란의 MVTecAD_colab과 동일한 주소입니다.
# ++ 만약에 train의 라벨이 필요하신 분들은 각 클래스의 train 폴더의 data.csv를 사용하시기 바랍니다.
print(train.keys())
print(val.keys())
print(test.keys())