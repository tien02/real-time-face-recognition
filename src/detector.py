from facenet_pytorch import MTCNN

class myMTCNN(MTCNN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def forward(self, img):
        # Detect faces
        batch_boxes, batch_probs, batch_points = self.detect(img, landmarks=True)
        # Select faces
        if not self.keep_all:
            batch_boxes, batch_probs, batch_points = self.select_boxes(
                batch_boxes, batch_probs, batch_points, img, method=self.selection_method
            )
        # Extract faces
        faces = self.extract(img, batch_boxes, None)

        return faces, batch_probs, batch_boxes, batch_points