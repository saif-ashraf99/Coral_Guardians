import os

def load_yolo_annotations(annotation_folder, image_folder):
    """
    Loads YOLO-format annotations. 
    YOLO text files have lines of the form: class_id center_x center_y width height
    (All normalized to [0,1].)
    """
    annotations_dict = {}
    for file_name in os.listdir(annotation_folder):
        if not file_name.endswith('.txt'):
            continue
        base_name = os.path.splitext(file_name)[0]
        image_path = os.path.join(image_folder, base_name + '.jpg')
        txt_path = os.path.join(annotation_folder, file_name)

        if not os.path.exists(image_path):
            continue

        with open(txt_path, 'r') as f:
            lines = f.readlines()
        bboxes = []
        for line in lines:
            class_id, cx, cy, w, h = line.strip().split(' ')
            class_id = int(class_id)
            cx, cy, w, h = float(cx), float(cy), float(w), float(h)
            bboxes.append((class_id, cx, cy, w, h))
        annotations_dict[image_path] = bboxes
    return annotations_dict
