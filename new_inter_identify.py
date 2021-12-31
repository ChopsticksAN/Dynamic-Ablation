from __future__ import print_function, division, absolute_import
import argparse
import os

from PIL import Image
from numpy import *
import torch
import cv2
import random
from tqdm import tqdm
import torchvision.transforms as transforms

import sys
sys.path.append('.')
import pretrainedmodels
import pretrainedmodels.utils as utils
from IPython import embed
from new_help_cut_noise import *
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries


CONFIG_repeat_number = 3
CONFIG_itr_number = 1000

CONFIG_SP_NUM = 1000

CONFIG_cut_rate = 0.97
CONFIG_step_rate = 0.3

CONFIG_center_numbers = [9, 16, 25]
#CONFIG_center_numbers = [4, 9, 16]
CONFIG_top_number = 3


model_names = sorted(name for name in pretrainedmodels.__dict__
    if not name.startswith("__")
    and name.islower()
    and callable(pretrainedmodels.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--arch', '-a', metavar='ARCH', default='nasnetalarge',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: nasnetalarge)',
                    nargs='+')
parser.add_argument('--path_img', type=str, default='data/cat.jpg')


def get_imagenet_synsets() :
    # Load Imagenet Synsets
    with open('data/imagenet_synsets.txt', 'r') as f:
        synsets = f.readlines()

    # len(synsets)==1001
    # sysnets[0] == background
    synsets = [x.strip() for x in synsets]
    splits = [line.split(' ') for line in synsets]
    key_to_classname = {spl[0]:' '.join(spl[1:]) for spl in splits}

    with open('data/imagenet_classes.txt', 'r') as f:
        class_id_to_key = f.readlines()

    class_id_to_key = [x.strip() for x in class_id_to_key]
    return class_id_to_key, key_to_classname

def cvimg_to_tensor(cvImg, model) :

    input_data = Image.fromarray(cvImg.astype(np.uint8))
    tf_img = utils.TransformImage(model)
    input_data = tf_img(input_data)      # 3x299x299
    input_data = input_data.unsqueeze(0) # 1x3x299x299
    image_tensor = torch.autograd.Variable(input_data)
    return image_tensor

def tensor_to_cvimg(img_tensor) : 
    result = img_tensor[0].numpy().transpose(1, 2, 0)
    result = np.round(result* 255).astype('uint8')
    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    return result

def make_prediction(model, img_tensor) :

    gpu = 0
    device = torch.device(
            f'cuda:{gpu}'
            if torch.cuda.is_available() and gpu is not None
            else 'cpu')
    model.to(device)
    #embed(header = 'device')
    img_tensor = img_tensor.to(device)

    class_id_to_key, key_to_classname = get_imagenet_synsets()
    # Make predictions
    output = model(img_tensor)                       # size(1, 1000)
    #L = sorted(output[0])
    softmax = torch.nn.Softmax(dim = 0)
    L = softmax(output.data.squeeze()).numpy().tolist()
    #max, argmax = output.data.squeeze().max(0)
    #class_id = argmax.item()
    #print(class_id)
    #class_key = class_id_to_key[class_id]
    #classname = key_to_classname[class_key]
    pred_list = []
    for idx, item in enumerate(L) :
        #if item > CONFIG_classification_thr :
        D = {'idx':idx, 'name':key_to_classname[class_id_to_key[idx]], 'conf':round(item, 3)}
        pred_list.append(D)


    #embed(header = "softmax")
    #print(classname) 
    return pred_list

def init_map(MAP_shape, disc_flag = True) :

    if disc_flag == True : 
        MAP = np.zeros(MAP_shape)
    elif disc_flag == False :
        MAP = np.ones(MAP_shape)

    return MAP

def get_MAP(MAP_shape, cand_list, slic_pos, disc_flag = True) :
    MAP = init_map(MAP_shape, disc_flag)
    for num in cand_list :
        if disc_flag == True : 
            MAP[slic_pos['dx'][num], slic_pos['dy'][num]] = 1
        else :
            MAP[slic_pos['dx'][num], slic_pos['dy'][num]] = 0
    return MAP

def slic_superpixel(cvImg) :
    global CONFIG_SP_NUM
    mask = slic(cvImg, n_segments = CONFIG_SP_NUM, compactness = 15)
    slic_Img = np.uint8(mark_boundaries(cvImg, mask, color = (1, 0, 0)) * 255)
    slic_Img = cv2.cvtColor(slic_Img, cv2.COLOR_RGB2BGR)
    cv2.imwrite('slic.jpg', slic_Img)

    n_seg = 0
    slic_pos = {'dx' : {}, 'dy' : {}}
    for i in range(mask.shape[0]) :
        for j in range(mask.shape[1]) :
            num = mask[i][j]
            n_seg = max(n_seg, num + 1)
            if num in slic_pos['dx'].keys() :
                slic_pos['dx'][num].append(i)
                slic_pos['dy'][num].append(j)
            else : 
                slic_pos['dx'][num] = [i]
                slic_pos['dy'][num] = [j]


    dx = [-1, 0, 1, 0]
    dy = [0, 1, 0, -1]
    edgeList = []
    slic_adj_list = {}
    for x in range(mask.shape[0]) :
        for y in range(mask.shape[1]) :
            for d in range(4) :
                tx = x + dx[d]
                ty = y + dy[d]
                if tx < 0 or tx >= mask.shape[0] or ty < 0 or ty >= mask.shape[1]: continue
                if mask[x][y] == mask[tx][ty] : continue
                if (mask[x][y], mask[tx][ty]) in edgeList : continue
                if (mask[tx][ty], mask[x][y]) in edgeList : continue
                edgeList.append((mask[x][y], mask[tx][ty]))
                slic_adj_list = build_edge(slic_adj_list, mask[x][y], mask[tx][ty])
                slic_adj_list = build_edge(slic_adj_list, mask[tx][ty], mask[x][y])


    return n_seg, mask, slic_pos, slic_adj_list


def Check_Top(top_number, pred_list, GT_idx, disc) :
    if disc == True :
        for item in pred_list[:top_number] :
            if item['idx'] == GT_idx :
                return True
        return False
    else :
        for item in pred_list[:top_number] :
            if item['idx'] == GT_idx :
                return False
        return True

def Update_Centers(MAP_shape, center_list, step_rate, R2_bound) :
    new_center_list = []
    sumR2 = 0
    pw = 1
    for center in center_list :
        x = center[0]
        y = center[1]
        r = center[2]
        dx = int(random.randint(-MAP_shape[0] + 1, MAP_shape[0] - 1) * step_rate)
        dy = int(random.randint(-MAP_shape[1] + 1, MAP_shape[1] - 1) * step_rate)
        dr = random.uniform(-(R2_bound ** 0.5), (R2_bound ** 0.5)) * step_rate
        #dr = 0
        #dr = random.uniform(-(min(MAP_shape[0], MAP_shape[1])), (min(MAP_shape[0], MAP_shape[1]))) * step_rate * 0.1
        dr = random.uniform(-r, r) * step_rate
        #dr = random.uniform(0, (min(MAP_shape[0], MAP_shape[1]))) * step_rate
        new_x = min(MAP_shape[0] - 1, max(0, x + dx))
        new_y = min(MAP_shape[1] - 1, max(0, y + dy))
        new_r = max(0.1, r + dr)
        #new_r = pw ** 2
        sumR2 += new_r * new_r
        new_center_list.append([new_x, new_y, new_r])
        pw = pw + 1
    for center in new_center_list :
        center[2] = (center[2] / (sumR2 ** 0.5)) * (R2_bound ** 0.5)

    return new_center_list

def get_cand_list(center_list, slic_pos, slic_number) :
    cand_list = []
    for num in range(slic_number) : 
        node_sp = (mean(slic_pos['dx'][num]), mean(slic_pos['dy'][num]))
        for center in center_list :
            node_center = (center[0], center[1])
            if Euclidean_distance(node_sp, node_center) <= center[2] :
                if num not in cand_list : 
                    cand_list.append(num)
    return cand_list 
    
def l0_boundary(cvImg, model, label = -1) :

    global CONFIG_step_rate
    global CONFIG_cut_rate
    global CONFIG_top_number
    total_center_list = []
    min_attribution_area = 10000000

    slic_number, slic_mask, slic_pos, slic_adj_list = slic_superpixel(cvImg)
    '''
    hd = input("yes or no : ")
    if hd == 'no' :
        return 0, 0, 0, 'skip'
    '''


    global CONFIG_repeat_number

    img_tensor = cvimg_to_tensor(cvImg, model)

    __mask = np.zeros((cvImg.shape[0], cvImg.shape[1]))
    mask = np.zeros((cvImg.shape[0], cvImg.shape[1]))

    GT_idx = -1

    text = ""
    circle_img = copy.deepcopy(cvImg)
    for ITR in range(CONFIG_repeat_number) :

        center_number = CONFIG_center_numbers[ITR]

        print("Round " + str(ITR))

        if ITR % 2 == 0 : 
            disc = False
            CONFIG_top_number = 1
        else :
            disc = False
            CONFIG_top_number = 1

        _mask = np.zeros((cvImg.shape[0], cvImg.shape[1]))
        global CONFIG_itr_number
        
        if GT_idx == -1 : 
            _pred_list = make_prediction(model, img_tensor)
            pred_list = sorted(copy.deepcopy(_pred_list), key = lambda e:e.__getitem__('conf'), reverse = True)
            print(pred_list[:10])
            GT_idx = int(input("Please input the category 'idx' to follow : "))
            #GT_idx = pred_list[0]['idx']
            GT_name = pred_list[0]['name']
            GT_thr = _pred_list[GT_idx]['conf']
        
        if label != -1 : 
            if GT_idx != label :
                return -1, -1, -1, "skip", -1
            GT_idx = label


        MAP_shape = cvImg.shape
        MAP = init_map(MAP_shape, disc_flag = disc)

        R_bound = (MAP_shape[0] * MAP_shape[1])
        
        center_list = []
        sq = int(center_number ** 0.5)
        for dx in range(sq) :
            for dy in range(sq) :
                x = dx * (cvImg.shape[0] // sq) + (cvImg.shape[0] // (sq* 2))
                y = dy * (cvImg.shape[1] // sq) + (cvImg.shape[1] // (sq* 2))
                r = sqrt(R_bound)
                center_list.append([x, y, r])

        
        last_center_list = []
        succ = 0
        col = 0
        for itr in range(CONFIG_itr_number) :
            CONFIG_step_rate_ =  max(0.05, CONFIG_step_rate * (1 - itr / CONFIG_itr_number))
            last_center_list = copy.deepcopy(center_list)
            
            center_list = Update_Centers(MAP_shape, center_list, CONFIG_step_rate_, R_bound)    
            cand_list = get_cand_list(center_list, slic_pos, slic_number)
            MAP = get_MAP(MAP_shape, cand_list, slic_pos, disc_flag = disc)     ###waiting...
        
            img_tensor_True = cvimg_to_tensor(cvImg * MAP, model)
            _pred_list_True = make_prediction(model, img_tensor_True)
            pred_list_True = sorted(copy.deepcopy(_pred_list_True), key = lambda e:e.__getitem__('conf'), reverse = True)

            if Check_Top(CONFIG_top_number, pred_list_True, GT_idx, disc) :
            #if Check_Top(5, pred_list_True, GT_idx, True) and Check_Top(5, pred_list_False, GT_idx, False) :
                print("Successful !")
                print(center_list)
                R_bound = R_bound * CONFIG_cut_rate
                text = "name  = '" + GT_name + "' conf = " + str(GT_thr)
                succ += 1
                if disc == True  : 
                    _mask += MAP[:, :, 0] * succ
                    min_attribution_area = min(min_attribution_area, np.sum(MAP[:, :, 0]))
                else :
                    _mask += (1 - MAP[:, :, 0]) * succ
                    min_attribution_area = min(min_attribution_area, np.sum(1 - MAP[:, :, 0]))

                '''
                if itr >= 400 : 
                    col = min(255, col + 30)
                    for idx, center in enumerate(reversed(center_list)) :
                        if disc == True :
                            C = (255, 255 - col, 0) 
                        else :
                            C = (255 - col, 255, 0)
                        if center[2] > 20 : 
                            circle_img = cv2.circle(circle_img, (center[1], center[0]), int(2), C, 2)
                    _circle_img = cv2.cvtColor(copy.deepcopy(circle_img), cv2.COLOR_RGB2BGR)
                    #cv2.imwrite("./circle_img.png", _circle_img)
                    #embed(header = "cv")
                '''
            else :
                '''
                if itr / CONFIG_itr_number < random.uniform(0.0, 1.0) : 
                    succ += 1
                    print("fake Successful !")
                    #print(center_list)
                    #R_bound = R_bound * CONFIG_cut_rate
                    text = "name  = '" + GT_name + "' conf = " + str(GT_thr)
                else :
                '''
                print("Failed !")
                center_list = copy.deepcopy(last_center_list)


            if itr % 50 == 0 : 
                cand_list = get_cand_list(last_center_list, slic_pos, slic_number)
                print("itr-" + str(itr) + " : There are " + str(len(cand_list)) + " superpixels left !")

        total_center_list += last_center_list
        
        #_mask = (_mask - np.min(_mask)) / (np.max(_mask) - np.min(_mask))
        __mask = __mask + _mask
        mask = (__mask - np.min(__mask)) / (np.max(__mask) - np.min(__mask))
        heat_map = cv2.applyColorMap(np.uint8(255 * (1 - mask)), cv2.COLORMAP_JET)

        heat_map = np.float32(heat_map)/ 255
        image = (cvImg - np.min(cvImg)) / (np.max(cvImg) - np.min(cvImg))
        cam = heat_map + np.float32(image)
        cam = cam / np.max(cam)
        result = np.uint8(255 * cam)
        show_result(cvImg, heat_map, mask, text, 'tmp.png')

        '''
        for idx, center in enumerate(reversed(last_center_list)) :
            if center[2] > 20 :
                if disc == True :
                    C = (0, 255, 0)
                else :
                    C = (255, 0, 0) 
                circle_img = cv2.circle(circle_img, (center[1], center[0]), int(center[2]), C, 2)
        _circle_img = cv2.cvtColor(copy.deepcopy(circle_img), cv2.COLOR_RGB2BGR)
        cv2.imwrite("./circle_img.png", _circle_img)
        '''

    #result = cvImg * MAP
    return result, heat_map, mask, text, min_attribution_area

def show_result(cvImg, heat_map, mask, text, img_name) :
    heat_map = cv2.resize(heat_map, (cvImg.shape[1], cvImg.shape[0]))
    image = (cvImg - np.min(cvImg)) / (np.max(cvImg) - np.min(cvImg))
    cam = heat_map + np.float32(image)
    cam = cam / np.max(cam)
    result = np.uint8(255 * cam)

    mask = cv2.resize(mask, (cvImg.shape[1], cvImg.shape[0]))
    cvImg = cv2.cvtColor(cvImg, cv2.COLOR_RGB2BGR)
    cvImg = cv2.cvtColor(cvImg, cv2.COLOR_BGR2BGRA)
    b, g, r, a = cv2.split(cvImg)
    zero = np.zeros(mask.shape)
    mask = np.where(mask < 0.8, zero, mask)
    a = np.uint8(mask * 255)
    #cvImg = cv2.merge((b, g, r, a))
    for i in range(3) :
        cvImg[:, :, i] = cvImg[:, :, i] * mask

    result_img = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    #part_img= cv2.cvtColor(cvImg, cv2.COLOR_RGB2BGR)
    #result_img = cv2.putText(result_img, text, (20, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
    cv2.imwrite(os.path.join(output_dir, "heat_" + img_name), result_img)
    #cvImg = cv2.putText(cvImg, text, (40, 50), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 0), 1.2)
    cv2.imwrite(os.path.join(output_dir, "trans_" + img_name), cvImg)

if __name__ == '__main__':
    global args
    args = parser.parse_args()

    for arch in args.arch:
        # Load Model

        img_path = args.path_img
        images_dir = "Imagenet2012/IMAGENET_VAL/ILSVRC2012_img_val/"
        txt_dir = "Imagenet2012/IMAGENET_VAL/val.txt"
        #images_dir = "/home/syc/imagenetbenchmark/test500"
        #txt_dir = "/home/syc/imagenetbenchmark/test500_label.txt"
        net_name = arch
        output_dir = "./selected/" 

        files = os.listdir(images_dir)
        files.sort()
        
        label_dict = {}
        with open(txt_dir, 'r') as file_to_read :
            for i in range(50000) :
                line = file_to_read.readline()
                label_dict[line.split(' ')[0]] = int(line.split(' ')[1])

        vis = np.zeros(50000)

        T = 1001
        for img_name_index in range(0, 50000) :

            model = pretrainedmodels.__dict__[arch](num_classes=1000,
                                                    pretrained='imagenet')
            model.eval()
            img_name = files[img_name_index]
            #img_name = "cat_dog.jpg"
            #img_name = "ILSVRC2012_val_000" + test_str + ".JPEG"
            img_name = "ILSVRC2012_val_00000013.JPEG"
            img_path = os.path.join(images_dir, img_name)
            print(img_path)
            load_img = utils.LoadImage()
            cvImg = np.array(load_img(img_path), dtype = 'uint8')
            #resized_Img = cvImg
            resized_Img = cv2.resize(copy.deepcopy(cvImg), (cvImg.shape[1] // 1, cvImg.shape[0] // 1))

            
            #resized_Img = cv2.resize(copy.deepcopy(cvImg), (100, 100))
            print('label = ' + str(label_dict[img_name]))
            result_resized_img, heat_map, mask, text, min_attribution_area = l0_boundary(resized_Img, model, label_dict[img_name])

            if text == "skip" : continue

            
            #file_handle = open(output_dir + "area.txt", mode = "a")
            #area_rate = min_attribution_area / (cvImg.shape[0] * cvImg.shape[1])

            #file_handle.write(img_name + ' ' + str(l) + ' ' + str(min_attribution_area) + ' ' + str(area_rate) + '\n')

            img_name = img_name[:-4] + ".png"
            show_result(cvImg, heat_map, mask, text, img_name)
            break






















