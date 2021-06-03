import os
def convert(filename,output, prefix=None):
    with open(output,'w') as output:
        with open(filename) as file:
            lines = file.readlines()
            for line in lines:
                line = line[:-1]
                res = line
                if prefix is not None:
                    res = os.path.join(prefix, line[line.rfind('\\')+1:])
                with open(line[:-4] + 'org.txt') as txt:
                    bboxs = txt.readlines()
                    for bbox in bboxs:
                        bbox = bbox[:-1]
                        bbox = bbox.split(' ')
                        bbox = list([int(i) for i in bbox])
                        x, y, w, h = bbox[3:]
                        res = res + ' ' + str(x) + ',' + str(y) + ',' + str(x + w) + ',' + str(y + h) + ',' + str(
                            bbox[0])
                output.write(res+'\n')

if __name__ == '__main__':
    filename = r'E:\DataSet\GJDataSet\overpass\train.txt'
    output = r'E:\DataSet\GJDataSet\overpass\train_of.txt'
    prefix= r'E:\DataSet\GJDataSet\overpass'
    convert(filename, output, prefix)