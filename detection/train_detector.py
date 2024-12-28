import os
import subprocess

def train_yolov5(
    data_yaml='data.yaml',
    imgsz=640,
    batch_size=8,
    epochs=50,
    device='0',
    project='runs/train',
    name='coral_model'
):
    """
    Train YOLOv5 using the official train.py script.
    
    - data_yaml: Path to a data configuration file describing train/val/test sets and class names.
    - imgsz: Image size for training.
    - batch_size: Batch size for training.
    - epochs: Number of training epochs.
    - device: GPU device (e.g., '0') or CPU ('cpu').
    - project: Folder to store training runs.
    - name: Name of the run.
    """
    yolov5_path = './yolov5'
    command = [
        'python', os.path.join(yolov5_path, 'train.py'),
        '--data', data_yaml,
        '--imgsz', str(imgsz),
        '--batch', str(batch_size),
        '--epochs', str(epochs),
        '--device', device,
        '--project', project,
        '--name', name
    ]
    subprocess.run(command)

if __name__ == '__main__':
    train_yolov5()
