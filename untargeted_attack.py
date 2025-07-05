import torchvision.transforms as transforms
import torchvision.models as torch_models
import numpy as np
import torch
import os
from utils import    get_label
from utils import valid_bounds
from PIL import Image
from torch.autograd import Variable
import time
from GSBA import GSBA



    
##############################################################################
torch.manual_seed(992)
torch.cuda.manual_seed(992)
np.random.seed(992)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
##############################################################################



num_img = 1000
iteration =130                      # Enter the maximum number of iterations
query_budget = 30010                # Ener the query budget
model_arc = 'resnet50'              # Enter 'resnet50' or 'resnet101' or 'vgg16' or 'ViT, DEiTB
top_k = 4

print('top-k:', top_k)
print('Number of iterations:', iteration)

# Models 
if model_arc == 'resnet50':
    net = torch_models.resnet50(pretrained=True)
elif model_arc == 'resnet101':
    net = torch_models.resnet101(pretrained=True)
elif model_arc == 'vgg16':
    net = torch_models.vgg16(pretrained=True) 
else:
    print('Not implemented for the given classifier')           
net = net.to(device)
net.eval()
        
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225] 
            
                    
all_norms = np.ones((num_img, iteration+1))*1000
all_queries = np.ones((num_img, iteration+1))*100000
image_iter =0 

for image_iter1 in range(1, 10000): 
    if image_iter>=num_img:
        break
    if len(str(image_iter1))==1:
        temp = "000"+ str(image_iter1)
    if len(str(image_iter1))==2:
        temp = "00"+ str(image_iter1)
    if len(str(image_iter1))==3:
        temp = "0"+ str(image_iter1) 
    if len(str(image_iter1))==4:
        temp =  str(image_iter1)
    img_name = "ILSVRC2012_val_0000" + temp + ".JPEG"
    # print('figure_num', temp)
    img_path = "path/to/image"

    t11 = time.time()
    
    im_orig = Image.open(os.path.join(img_path, img_name))
    if np.array(im_orig).shape[-1]!=3:
        im_orig = im_orig.convert('RGB')
    im_sz = 224
    im_orig = transforms.Compose([transforms.Resize((im_sz, im_sz))])(im_orig)
    
    delta = 255
    lb, ub = valid_bounds(im_orig, delta)
        
    # Transform data
    im = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean = mean,
                                std = std)])(im_orig)
    
    
    lb = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])(lb)
    ub = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])(ub)
    
    lb = lb[None, :, :, :].to(device)
    ub = ub[None, :, :, :].to(device)
    
    x_0 = im[None, :, :, :].to(device)
    
    orig_label = torch.argmax(net.forward(Variable(x_0, requires_grad=True)).data).item()
    labels = open(os.path.join('synset_words.txt'), 'r').read().split('\n')
    str_label_orig = get_label(labels[np.int32(orig_label)].split(',')[0])
    
    ground_truth  = open(os.path.join('val.txt'), 'r').read().split('\n')
    
    ground_name_label = ground_truth[image_iter1-1]   #what is image_num
    ground_label_split_all =  ground_name_label.split
    
    ground_label_split =  ground_name_label.split()
    
    ground_label =  ground_name_label.split()[1]
    ground_label_int = int(ground_label)
        
    
    str_label_ground = get_label(labels[np.int32(ground_label)].split(',')[0])

    print(f'\nSource image {temp}:  Class ID: {ground_label}   Class Name: {str_label_ground}')

##############################################################################        
    
    
                
    if ground_label_int != int(orig_label) :
        print('Already missclassified ... Lets try another one!')
        
    else:    
        
    
        image_iter = image_iter + 1
        print('Image number good to go: ', image_iter,'/', num_img)

        print(f'############################# topK:{top_k} #####################################')
        print(f'Start: non-targeted attck will be run for {iteration} iterations')
        print(f'############################# {model_arc} ####################################')
    
    
        t3 = time.time()
        attack = GSBA(net, x_0, mean, std, lb, ub, 
                                    iteration=iteration, query_budget=query_budget,
                                    top_k = top_k, device=device, 
                                    )
        x_adv, n_query, norms= attack.Attack()
        t4 = time.time()
        print(f'##################### End Itetations: queries:{n_query[-1]}  took {t4-t3:.3f} sec #########################')
        
        all_norms[image_iter-1][:len(norms)] = norms
        all_queries[image_iter-1][:len(n_query)] = n_query
    
        
folder = 'results_untargeted'

if not os.path.exists(folder):
    os.makedirs(folder)
    
np.savez(f'{folder}/Untargeted_{model_arc}_imgNum_{image_iter}_query_budget_{query_budget}_topK{top_k}',
        all_norms = all_norms,
        all_queries = all_queries)
    
    
