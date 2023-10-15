from ultralytics import YOLO

# Load a model
model = YOLO("best.pt")

custom_order = {
    1: 'Yellow',
    2: 'Blue',
    3: 'Red',
    4: 'Purple',
    5: 'Orange',
    6: 'Green',
    7: 'Brown',
    8: 'Black'
}

original_list ={
    0: 'Black',
    1: 'Blue',
    2: 'Purple',
    3: 'Yellow',
    4: 'Green',
    5: 'Brown',
    6: 'Red',
    7: 'Orange',
}

#getting the results for the image
results = model(source="predict/444.mp4", show=True, conf=0.85,save=True)
# set = []
# # Loop through the results and print  label names
# for r in results:
#    boxes=r.boxes.cpu().numpy()
#    boxData=boxes.data
#    sorted_list = sorted(boxData, key=lambda x: x[0])
#    for x in sorted_list:
#        set.append(int(x[-1]))
#
# # Create a mapping from the values in the given_list to the custom_order keys
# mapping = {value: key for key, value in custom_order.items()}
#
# # Map the given_list values to the custom_order keys
# finalResults = [mapping[original_list[given_value]] for given_value in set]
#
# print(finalResults)








