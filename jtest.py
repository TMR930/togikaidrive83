import json

json_open = open('json/yolov7tiny_coco_416x416.json', 'r')
json_load = json.load(json_open)

# print(json_load['mappings']['labels'])

labels = json_load['mappings']['labels']

print(labels[0])