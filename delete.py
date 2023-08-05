import os
sum = 0
path = 'data/database collage/detections/DB unified of friends/DB without augmentation'

for dir in os.listdir(path):
    for filename in os.listdir(os.path.join(path, dir)):
            sum += 1
print(sum)