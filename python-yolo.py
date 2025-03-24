from ultralytics import YOLO
import os

folder_path = 'C:\\Users\\Costco Markham East\\Desktop\\pythonProject\\datasets\\test\\images'

model = YOLO('broken_wire_1.pt')

for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)

    if os.path.isfile(file_path):
        try:
            results = model(file_path)
            for i, results in enumerate(results):
                print(f'Dsiplaying results for image{i + 1}')
                results.show()

            input(f'Press enter for next file...')
        except Exception as e:
            print(f'Could not open {filename}: {e}')
