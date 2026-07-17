import json
import glob
import cv2 as cv
import os


class tococo(object):
    def __init__(self, jpg_paths, label_path, save_path, img_id=0, ann_id=0):
        self.images = []
        self.categories = []
        self.annotations = []
        # list of image file paths
        self.jpgpaths = jpg_paths
        self.save_path = save_path
        self.label_path = label_path
        # classes can be configured as needed; here only one class is set
        self.class_ids = {'FH': 1}
        self.name = 'FH'
        self.coco = {}
        self.img_id = img_id
        self.ann_id = ann_id
        self.yolo_to_coco()  # perform the conversion

    def yolo_to_coco(self):
        # annid = 0  # annotation id (unique across the whole dataset)
        for num, jpg_path in enumerate(self.jpgpaths):  # iterate over each image

            # extract the current image name (without extension) from the path
            imgname = jpg_path.split('\\')[-1].split('.')[0]  # for Windows paths
            # imgname = jpg_path.split('/')[-1].split('.')[0]  # for Unix paths
            img = cv.imread(jpg_path)  # read the current image
            h, w = img.shape[:-1]  # get image height and width (default 640)
            self.images.append(self.get_images(imgname, h, w, self.img_id, jpg_path))  # update images list
            # read the YOLO-format labels from the .txt file
            with open(self.label_path + imgname + '.txt', 'r') as f:
                labels = f.readlines()

            for label in labels:  # iterate over each bounding box
                label = list(map(float, label.strip().split()))  # convert string to a list of floats
                if len(label) >= 5:  # ensure at least 5 columns (YOLO format)
                    class_id, x_center, y_center, w_, h_ = label[:5]
                # convert center coordinates to top-left coordinates and denormalize
                class_id = class_id + 1
                px = (x_center - w_ / 2) * w
                py = (y_center - h_ / 2) * h
                pw = w_ * w
                ph = h_ * h
                box = [px, py, pw, ph]
                # update the annotations list
                self.annotations.append(self.get_annotations(box, self.img_id, self.ann_id, int(class_id)))
                self.ann_id = self.ann_id + 1  # annotation id must increment across the dataset
            self.img_id = self.img_id + 1
        # update category list
        self.categories.append(self.get_categories(1, self.name))  # single-class dataset, hence written this way
        # store everything into the COCO dictionary
        self.coco["images"] = self.images
        self.coco["categories"] = self.categories
        self.coco["annotations"] = self.annotations

    def get_images(self, filename, height, width, image_id, jpg_path):
        image = {}
        image["height"] = height
        image['width'] = width
        image["id"] = image_id
        image["img_path"] = jpg_path
        # file name with extension
        image["file_name"] = filename + '.png'
        return image

    def get_categories(self, class_id, name):
        category = {}
        category["supercategory"] = "Positive Cell"  # parent category
        category['id'] = class_id
        category['name'] = name
        return category

    def get_annotations(self, box, image_id, ann_id, class_ids):
        annotation = {}
        w, h = box[2], box[3]
        area = w * h
        annotation['segmentation'] = [[]]
        annotation['iscrowd'] = 0
        annotation['image_id'] = image_id
        annotation['bbox'] = box
        annotation['area'] = float(area)
        annotation['category_id'] = class_ids
        annotation['id'] = ann_id
        return annotation

    def save_json(self):
        label_dic = self.coco
        # serialize the Python object (dict/list) into a JSON string
        instances_train = json.dumps(label_dic)
        # set the save path and file name
        f = open(os.path.join(save_path + '/instances_val.json'), 'w')
        f.write(instances_train)
        f.close()

    def get_coco(self):
        return self.coco, self.img_id, self.ann_id


if __name__ == '__main__':

    print(os.getcwd())
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # get the list of all image paths under the train/val folder
    img_paths = glob.glob(current_dir + '/val/*.png')
    # existing label file directory
    label_path = current_dir + '/labels/val/'
    # output directory
    save_path = current_dir + '/annotations'
    c = tococo(img_paths, label_path, save_path)  # create the YOLO-to-COCO converter instance
    c.save_json()  # convert and save
