from keras.callbacks import ReduceLROnPlateau
from keras.layers import (Input, Conv2D, BatchNormalization, ZeroPadding2D,
                          MaxPooling2D, Activation, Dense, Dropout, Flatten)
from keras.models import Model
from keras import optimizers
from model import BaseModel
from train import train

MODEL_NAME = 'VGG16' # This should be modified when the model name changes.

class VGG16(BaseModel):
    '''
    1. ZeroPadding2D (2, 2)
    2. (3X3 Conv2D 64) X 2 + maxpool
    3. (3X3 Conv2D 128) X 2 + maxpool
    4. (3X3 Conv2D 256) X 3 + maxpool
    5. (3X3 Conv2D 512) X 3 + maxpool
    6. (3X3 Conv2D 512) X 3 + maxpool
    7. (FC 256 + Dropout(0.5)) X 2
    8. FC 10 + Softmax

    '_build()' is only modified when the model changes.

    HowToUse:
        model = VGG16(PATH_FOR_WEIGHTS)
        * all funtionalities are written in BaseModel.py
    '''
    def __init__(self):
        callbacks = [ReduceLROnPlateau(monitor = 'val_loss', factor = 0.1,
                                       patience = 5, verbose = 1)]
        optimizer = optimizers.SGD(lr=0.01, momentum=0.9, decay=1e-04)
        BaseModel.__init__(self, model = self._build(), optimizer = optimizer,
                           callbacks = callbacks)

    def _build(self):
        x = Input(shape = (28, 28, 1))
        y = ZeroPadding2D(padding = (2, 2))(x) # matching the image size of CIFAR-10

        y = self._multi_conv_pool(y, 64, 2) # 32x32
        y = self._multi_conv_pool(y, 128, 2) # 16x16
        y = self._multi_conv_pool(y, 256, 3) # 8x8
        y = self._multi_conv_pool(y, 512, 3) # 4x4
        y = self._multi_conv_pool(y, 512, 3) # 2x2
        y = Flatten()(y)
        y = Dense(units = 256, activation='relu')(y) # original paper suggests 4096 FC
        y = Dropout(0.5)(y)
        y = Dense(units = 256, activation='relu')(y)
        y = Dropout(0.5)(y)
        y = Dense(units = 10, activation='softmax')(y)

        return Model(x, y, name = MODEL_NAME)

    def _multi_conv_pool(self, x, output_channel, n):
        # The training is regularized by l2 weight decay(5e-4) in the original paper
        # BN is used here instead of weight decay
        y = x
        for _ in range(n):
            y = Conv2D(output_channel, (3, 3), padding = 'same')(y)
            y = BatchNormalization()(y)
            y = Activation('relu')(y)
        y = MaxPooling2D(strides = (2, 2))(y)
        return y

def main():
    '''
    Train the model defined above.
    '''
    model = VGG16()
    train(model, MODEL_NAME)

if __name__ == '__main__':
    main()