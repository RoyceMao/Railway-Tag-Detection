import cv2


def get_data(input_path):
    found_bg = False
    all_imgs = {}
    classes_count = {}
    class_mapping = {}

    with open(input_path, 'r') as f:
        for line in f:
            line_split = line.strip().split(',')
            (filename, x1, y1, x2, y2, class_name) = line_split

            if class_name not in classes_count:
                classes_count[class_name] = 1
            else:
                classes_count[class_name] += 1

            if class_name not in class_mapping:
                if class_name == 'bg' and not found_bg:
                    print('Found class name with special name bg. Will be treated as a'
                          ' background region (this is usually for hard negative mining).')
                    found_bg = True
                class_mapping[class_name] = len(class_mapping)

            if filename not in all_imgs:
                all_imgs[filename] = {}

                img = cv2.imread(filename)
                (rows, cols) = img.shape[:2]
                all_imgs[filename]['filepath'] = filename
                all_imgs[filename]['width'] = cols
                all_imgs[filename]['height'] = rows
                all_imgs[filename]['bboxes'] = []
                all_imgs[filename]['imageset'] = 'trainval'
                all_imgs[filename]['outer_boxes'] = []
                # if np.random.randint(0, 6) > 0:
                #     all_imgs[filename]['imageset'] = 'trainval'
                # else:
                #     all_imgs[filename]['imageset'] = 'test'

            all_imgs[filename]['bboxes'].append(
                {'class': class_name, 'x1': int(float(x1)), 'x2': int(float(x2)), 'y1': int(float(y1)),
                 'y2': int(float(y2))})

        all_data = []
        for key in all_imgs:
            x1s = [gt['x1'] for gt in all_imgs[key]['bboxes'][:3]]
            x2s = [gt['x2'] for gt in all_imgs[key]['bboxes'][:3]]
            y1s = [gt['y1'] for gt in all_imgs[key]['bboxes'][:3]]
            y2s = [gt['y2'] for gt in all_imgs[key]['bboxes'][:3]]
            all_imgs[key]['outer_boxes'].append({'class': 'tag', 'x1': min(x1s), 'x2': max(x2s),
                                                 'y1': min(y1s), 'y2': max(y2s)})
            if all_imgs[key]['bboxes'][3:6]:
                x1s = [gt['x1'] for gt in all_imgs[key]['bboxes'][3:6]]
                x2s = [gt['x2'] for gt in all_imgs[key]['bboxes'][3:6]]
                y1s = [gt['y1'] for gt in all_imgs[key]['bboxes'][3:6]]
                y2s = [gt['y2'] for gt in all_imgs[key]['bboxes'][3:6]]
                all_imgs[key]['outer_boxes'].append({'class': 'tag', 'x1': min(x1s), 'x2': max(x2s),
                                                     'y1': min(y1s), 'y2': max(y2s)})

            if all_imgs[key]['bboxes'][3:6] and all_imgs[key]['bboxes'][6:]:
                x1s = [gt['x1'] for gt in all_imgs[key]['bboxes'][6:]]
                x2s = [gt['x2'] for gt in all_imgs[key]['bboxes'][6:]]
                y1s = [gt['y1'] for gt in all_imgs[key]['bboxes'][6:]]
                y2s = [gt['y2'] for gt in all_imgs[key]['bboxes'][6:]]
                all_imgs[key]['outer_boxes'].append({'class': 'tag', 'x1': min(x1s), 'x2': max(x2s),
                                                     'y1': min(y1s), 'y2': max(y2s)})
            all_data.append(all_imgs[key])

        # make sure the bg class is last in the list
        if found_bg:
            if class_mapping['bg'] != len(class_mapping) - 1:
                key_to_switch = [key for key in class_mapping.keys() if class_mapping[key] == len(class_mapping) - 1][0]
                val_to_switch = class_mapping['bg']
                class_mapping['bg'] = len(class_mapping) - 1
                class_mapping[key_to_switch] = val_to_switch

        return all_data, classes_count, class_mapping
