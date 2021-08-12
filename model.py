import paddle
import paddle.nn as nn
from paddle.vision.models import resnet34


class Model(nn.Layer):
    """
    simply create a 2-branch network, and concat global pooled feature vector.
    each branch = single resnet34
    """

    def __init__(self):
        super(Model, self).__init__()
        self.fundus_branch = resnet34(pretrained=True, num_classes=0)  # remove final fc
        self.oct_branch = resnet34(pretrained=True, num_classes=0)  # remove final fc
        self.decision_branch = nn.Linear(512 * 1 * 2, 3)  # ResNet34 use basic block, expansion = 1

        # replace first conv layer in oct_branch
        self.oct_branch.conv1 = nn.Conv2D(256, 64,
                                          kernel_size=7,
                                          stride=2,
                                          padding=3,
                                          bias_attr=False)

    def forward(self, fundus_img, oct_img):
        b1 = self.fundus_branch(fundus_img)
        b2 = self.oct_branch(oct_img)
        b1 = paddle.flatten(b1, 1)
        b2 = paddle.flatten(b2, 1)
        logit = self.decision_branch(paddle.concat([b1, b2], 1))

        return logit