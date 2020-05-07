DATASET = 'omniglot'
FCE = False
DISTANCE = 'cosine' # cosine or l2
N_TRAIN = 1
N_TEST = 1
K_TRAIN = 5
K_TEST = 20
Q_TRAIN = 15
Q_TEST = 1
LSTM_LAYERS = 1
UNROLLING_STEPS = 2

for obj in classes_shots.keys():
            # add the query image to each object 
            images.append({
                        'dataset': dataset,
                        'class_name': obj,
                        'file_name': next(iter(query[obj].keys())),
                        'filepath': query[obj][q_img]
                    })
                    
            for image in classes_shots[obj]:
                for i in range(classes_shots[obj][image]):
                    images.append({
                        'dataset': dataset,
                        'class_name': obj,
                        'file_name': image,
                        'filepath': os.path.join("data", dataset, "images" , obj, image)
                    })
