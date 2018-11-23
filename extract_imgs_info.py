from bs4 import BeautifulSoup
import os
import glob


def extract_imgs_infos(annots_dir, imgs_dir):
    """
    抽取图片的标注信息，存入txt
    :param annots_dir: 标注文件目录
    :param imgs_dir: 图片目录
    :return:
    """
    annots_file_name = glob.glob1(annots_dir, '*.xml')
    for annot_file_name in annots_file_name:
        with open(os.path.join(annots_dir, annot_file_name)) as f:
            text = f.read()

        objects = []
        img_path = os.path.join(imgs_dir, annot_file_name.replace('xml', 'png'))
        soup = BeautifulSoup(text, 'lxml')
        for obj in soup.find_all('object'):
            cls_name = obj.find('name').text
            xmin = obj.find('xmin').text
            ymin = obj.find('ymin').text
            xmax = obj.find('xmax').text
            ymax = obj.find('ymax').text
            objects.append(','.join([img_path, xmin, ymin, xmax, ymax, cls_name]) + '\n')

        with open('img_infos_voc.txt', 'a+', encoding='utf-8') as f:
            f.writelines(objects)


if __name__ == '__main__':
    extract_imgs_infos('00020_annotated_num_120/annotations', '00020_annotated_num_120/images')
