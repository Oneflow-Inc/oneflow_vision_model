import os
import argparse
parser = argparse.ArgumentParser(description='flags for detect video')
parser.add_argument(
    '-dataset_path',
    '--dataset_path',
    type = str,
    required = True
)
args = parser.parse_args()

def generate_txt(path, output):
    with open(output,'w') as output:
        for filename in os.listdir(path):
            if not filename.endswith('.jpg'):
                continue
            res = os.path.join(path, filename)
            with open(os.path.join(path, filename[:-4] + 'org.txt')) as txt:
                bboxs = txt.readlines()
                for bbox in bboxs:
                    bbox = bbox[:-1]
                    bbox = bbox.split(' ')
                    bbox = list([int(i) for i in bbox])
                    x, y, w, h = bbox[3:]
                    bbox[0] = 0
                    res = res + ' ' + str(x) + ',' + str(y) + ',' + str(x + w) + ',' + str(y + h) + ',' + str(
                        bbox[0])
            output.write(res + '\n')


if __name__ == '__main__':
    dataset_path = 'dataset'
    train_path = os.path.join(args.dataset_path, 'train')
    train_txt = os.path.join('data', 'dataset', 'train_of.txt')
    generate_txt(train_path, train_txt)
    train_path = os.path.join(args.dataset_path, 'val')
    train_txt = os.path.join('data', 'dataset', 'val_of.txt')
    generate_txt(train_path, train_txt)