import torch.nn as nn
import pretrainedmodels
class Image_Model(nn.Module):
    def __init__(self, model, visual_feature_size, num_labels=41, dropout_rate=0.1):
        super(Image_Model, self).__init__()
        modules = list(model.children())[:-1]
        self.model = nn.Sequential(*modules)
        # self.dropout_rate = nn.Dropout(dropout_rate)
        self._visual_feature_size = visual_feature_size
        # self.classification_layer = nn.Linear(model.last_linear.in_features, num_labels)
        # self.image_loss_fn = nn.BCEWithLogitsLoss()
    #
    # def fbeta(self,y_pred, y_true, thresh: float = 0.5, beta: float = 2, eps: float = 1e-9, sigmoid: bool = True):
    #     "Computes the f_beta between `preds` and `targets`"
    #     beta2 = beta ** 2
    #     if sigmoid: y_pred = y_pred.sigmoid()
    #     y_pred = (y_pred > thresh).float()
    #     y_true = y_true.float()
    #     TP = (y_pred * y_true).sum(dim=1)
    #     prec = TP / (y_pred.sum(dim=1) + eps)
    #     rec = TP / (y_true.sum(dim=1) + eps)
    #     res = (prec * rec) / (prec * beta2 + rec + eps) * (1 + beta2)
    #     return res.mean()
    #
    # def accuracy_thresh(self, y_pred, y_true, thresh: float = 0.5, sigmoid: bool = True):
    #     "Computes accuracy when `y_pred` and `y_true` are the same size."
    #     if sigmoid: y_pred = y_pred.sigmoid()
    #
    #     return ((y_pred > thresh).byte() == y_true.byte()).float().mean()

    def forward(self, x, inference=False):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        # classification_out = self.classification_layer(x)
        # if not inference:
        #     labels = input['labels']
        #     classification_loss = self.image_loss_fn(classification_out, labels)
        #     f1_score = self.fbeta(classification_out, labels)
        #     accuracy = self.accuracy_thresh(classification_out, labels)
        #     return classification_loss, f1_score, accuracy, classification_out
        return x
    @property
    def visual_feature_size(self) -> int:
        r"""
        Size of the channel dimension of output from forward pass. This
        property is used to create layers (heads) on top of this backbone.
        """
        return self._visual_feature_size