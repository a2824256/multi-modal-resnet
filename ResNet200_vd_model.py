import paddle
import paddle.nn as nn
from ppcls.arch.backbone.legendary_models.resnet import ResNet200_vd

class ResNet200_vd_model(nn.Layer):

    def __init__(self):
        super(ResNet200_vd_model, self).__init__()
        self.fundus_branch = ResNet200_vd(pretrained=True, class_num=0)
        self.oct_branch = ResNet200_vd(pretrained=True, class_num=0, input_image_channel=256)
        self.decision_branch = nn.Linear(2048 * 1 * 2, 1024)  # ResNet34 use basic block, expansion = 1
        self.output_branch = nn.Linear(1024, 3)  # ResNet34 use basic block, expansion = 1

    def forward(self, fundus_img, oct_img):
        b1 = self.fundus_branch(fundus_img)
        b2 = self.oct_branch(oct_img)
        b1 = paddle.flatten(b1, 1)
        b2 = paddle.flatten(b2, 1)
        logit = self.decision_branch(paddle.concat([b1, b2], 1))
        output = self.output_branch(logit)

        return output



# x1 = paddle.randn((1, 3, 512, 512), dtype='float32')
# x2 = paddle.randn((1, 256, 512, 512), dtype='float32')
# model = ResNet200_vd_model()
# output = model(x1, x2)
# print(output.shape)