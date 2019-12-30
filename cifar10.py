from scipy.misc import imsave
import numpy as np
import os

labels = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
pic_count = [1000,1000,1000,1000,1000,1000,1000,1000,1000,1000]

def unpickle(file):
    import pickle as pk
    fo = open(file, 'rb')
    dict = pk.load(fo,encoding='iso-8859-1')
    fo.close()
    return dict


for j in range(1, 6):
    dataName = "cifar-10-python/cifar-10-batches-py/data_batch_" + str(j)  
    Xtr = unpickle(dataName)
    print (dataName + " is loading...")

    for i in range(0, 10000):
        img = np.reshape(Xtr['data'][i], (3, 32, 32))  
        img = img.transpose(1, 2, 0)  
        label = labels[Xtr['labels'][i]]
        path = 'train_dataset/' + label
        if pic_count[Xtr['labels'][i]] == 0:
            continue
        picName = path + '/' + str(i + (j - 1)*10000) + '.jpg'
        imsave(picName, img)
        pic_count[Xtr['labels'][i]] = pic_count[Xtr['labels'][i]]-1
    print (dataName + " loaded.")

print ("test_batch is loading...")


testXtr = unpickle("cifar-10-python/cifar-10-batches-py/test_batch")
for i in range(0, 100):
    img = np.reshape(testXtr['data'][i], (3, 32, 32))
    img = img.transpose(1, 2, 0)
    picName = 'test_dataset/' + str(testXtr['labels'][i]) + '_' + str(i) + '.jpg'
    imsave(picName, img)
print ("test_batch loaded.")
