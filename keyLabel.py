class keyLabel:
    def __init__(self):
        self.label_to_int = {b:a for a, b in enumerate(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ*"))}
        self.int_to_label = {a:b for a, b in enumerate(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ*"))}

    def transform(self, labels):
        return [self.label_to_int[label] for label in labels]

    def inverse_transform(self, ints):
        return [self.int_to_label[i] for i in ints]