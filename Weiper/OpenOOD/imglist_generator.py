import os
import random
import time

def generate_imglist(k=3e5, classes=1000):
    """
    creates shuffled, balanced subset of /benchmark_imglist/imagenet/train_imagenet.txt
    """
    
    save_path = "/srv/data/manuelheurich/openood/benchmark_imglist/imagenet/dev_imagenet.txt"
    old_path = "/srv/data/manuelheurich/openood/benchmark_imglist/imagenet/train_imagenet.txt"
    read_path = "/srv/data/manuelheurich/openood/benchmark_imglist/imagenet/train_imagenet_shuffled.txt"
    
    # Shuffle the og file
    with open(old_path, 'r') as old:
        lines = old.readlines()
        random.shuffle(lines)
        with open(read_path, 'w') as new:
            new.writelines(lines)
        new.close()
    old.close()
    
    label_count = [0] * classes
    open(save_path, 'w').close()
    
    with open(save_path, 'w') as w:
        with open(read_path, 'r') as r:
            while True:
                line = r.readline()
                if not line:
                    r.close()
                    w.close()
                    return
            
                splits = line.split()
                label = splits[1]
                
                if label_count[int(label)] > int(k/classes):
                    continue
                else:
                    label_count[int(label)] += 1
                
                w.write(line)
                
                if sum(label_count) >= k:
                    r.close()
                    w.close()
                    return     


if __name__ == '__main__':
    start = time.time()
    generate_imglist()
    end = time.time()
    print(end - start)
    
    
'''
path="./data/images_classic/cinic/valid"
save_path="./data/benchmark_imglist/cifar10/val_cinic10.txt"
prefix="cinic/valid/"
category=["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
with open(save_path,'a') as f:
    for name in category:
        label=category.index(name)
        sub_path=path+'/'+name
        files=os.listdir(sub_path)
        for file in files:
            line=prefix+name+'/'+file+' '+str(label)+'\n'
            f.write(line)
    f.close()       
'''

# path="./data/images_classic/cifar100c"
# save_path="./data/benchmark_imglist/cifar100/test_cifar100c.txt"
# prefix="cifar100c/"
# files=os.listdir(path)
# with open(save_path,'a') as f:
#     for file in files:
#         splits=file.split("_")
#         label=(splits[1].split("."))[0]
#         line=prefix+file+" "+label+'\n'
#         f.write(line)
#     f.close()  

'''
path="./data/images_largescale/imagenet_v2"
save_path="./data/benchmark_imglist/imagenet/test_imagenetv2.txt"
prefix="imagenet_v2/"
with open(save_path,'a') as f:
    for i in range(0,1000):
        label=str(i)
        sub_path=path+'/'+label
        files=os.listdir(sub_path)
        for file in files:
            line=prefix+label+'/'+file+' '+label+'\n'
            f.write(line)
    f.close() 
'''