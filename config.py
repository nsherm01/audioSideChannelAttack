class config:
    img_size = (224, 224)
    num_classes = 100
    batch_size = 8
    weight_decay = 1e-8 # taken from the paper
    penalty = -3.30
    radius = 15
    lr = 5e-5 # taken from the paper
    epochs = 1