master_port = '2564'
gpus = 4
nodes = 1
node_rank = 0
world_size = 4 # gpus*nodes

ffpp_original_path = "/data1/chenpeng/data/FaceForensics++/sample/faces_13/original_sequences/youtube"
ffpp_fake_path = "/data1/chenpeng/data/FaceForensics++/sample/faces_13/manipulated_sequences"
celebdf_path = "/data5/tjh/standard_crop/Celeb-DFv2"
dfdc_path = "/data5/tjh/standard_crop/select_dfdc_test/videos"
dfdc_data_list =  "/data5/tjh/standard_crop/dfdc_test.txt"
#celeb-df
Deepfakes_path = f'{celebdf_path}'
original_path = f'{celebdf_path}'
youtube_path = f'{celebdf_path}'
split_path = '/data1/chenpeng/data/Celeb-DFv2/test.txt'

DETERMINSTIC = True
RNG_SEED = 20
batch_size = 32 #128
frame_nums = 10

compress = 'c23'    #choose c23 or c40

input_size = 224
#TODO: mtcnn is used for?
# detect_name = "mtcnn"
model_name = 'efficientnet-b4'
type_list = ['Deepfakes','Face2Face','FaceSwap','NeuralTextures','all']
type_list_short = ['df','f2f','fs','nt','all']
real_weight = 1


model_path = None
fnet_path = None

beta1 = 0.9
beta2 = 0.999
learning_rate = 0.001
metalr = 0.001
plr = 0.001
epochs = 200
step_size = 5
gamma = 0.1
weight_bce = 0.05
weight_ct = 1
# Misc
print_interval = 100
save_interval = 5
test_interval = 1
finetune_interval = 2
parallel = True
lamda = 0.01
alpha = 1
des = "train"
save_dir = f"result/output_{compress}_GCD_ablation/{compress}_{model_name}_{input_size}_{lamda}_{alpha}_{des}"