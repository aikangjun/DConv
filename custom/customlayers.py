from custom import *
import tensorflow as tf


class ConvOffset2D(layers.Layer):
    '''
    Deformable Convolution 可形变卷积,仅支持二维卷积
    '''

    def __init__(self,
                 kernel_size: tuple,
                 strides: tuple,
                 padding: str = 'SAME',
                 **kwargs):
        super(ConvOffset2D, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding

    def build(self, input_shape):
        self.conv_offset = layers.Conv2D(filters=input_shape[-1] * 2,
                                         kernel_size=self.kernel_size,
                                         strides=self.strides,
                                         padding=self.padding,
                                         trainable=True)
        self.built = True

    def to_bc_h_w_2(self, offset, inputs_shape):
        '''
        (b,h,w,2c) -> (b*c,h,w,2)
        :param offset:
        :param inputs_shape:
        :return:
        '''
        x = tf.transpose(offset, perm=[0, 3, 1, 2])
        x = tf.reshape(x, shape=(-1, inputs_shape[1], inputs_shape[2], 2))
        return x

    def to_bc_h_w(self, inputs, inputs_shape):
        '''
        (b,h,w,c) ->(b*c,h,w)
        :param inputs:
        :param inputs_shape:
        :return:
        '''
        x = tf.transpose(inputs, perm=[0, 3, 1, 2])
        x = tf.reshape(x, shape=(-1, inputs_shape[1], inputs_shape[2]))
        return x

    def get_coords(self, inputs, offsets):
        '''

        :param inputs: (b*c,h,w)
        :param offsets: (b*c,h,w,2)
        :return: (b*c,h*w,2)
        '''
        input_shape = tf.shape(inputs)
        bc = input_shape[0]
        h = input_shape[1]
        w = input_shape[2]
        offsets = tf.reshape(offsets, (bc, -1, 2))  # (b*c,h*w,2)
        grid = tf.meshgrid(
            tf.range(h), tf.range(w), indexing='ij'
        )
        grid = tf.stack(grid, axis=-1)  #
        grid = tf.cast(grid, 'float32')
        grid = tf.reshape(grid, (-1, 2))  # (h*w,2)
        grid = tf.expand_dims(grid, axis=0)  # (1,h*w,2)
        grid = tf.tile(grid, multiples=[bc, 1, 1])  # (b*c,h*w,2)
        coords = offsets + grid
        return coords

    def tf_repeat(self, a, repeats):
        a = tf.expand_dims(a, -1)
        a = tf.tile(a, [1, repeats])
        a = tf.reshape(a, shape=(-1))
        return a

    def tf_batch_map_coordinates(self, inputs, coords):
        '''
        映射坐标，得到新坐标的所有像素值，仅支持二维
        :param inputs: (b*c,h,w)
        :param coords: (b*c,h*w,2)
        :return:
        '''
        input_shape = tf.shape(inputs)
        bc = input_shape[0]
        h = input_shape[1]
        w = input_shape[2]
        hw = tf.shape(coords)[1]

        '''
        # 该段代码会导致不能计算梯度，原因是将tensor转为了numpy,需要换一种方式，对tensor的数据处理
        # tensorflow不能对切片元素赋值，pytorch则可以直接更改元素
        coords = coords.numpy()
        coords[..., 0] = tf.clip_by_value(coords[..., 0], clip_value_min=0, clip_value_max=tf.cast(w, tf.float32) - 1)
        coords[..., 1] = tf.clip_by_value(coords[..., 1], clip_value_min=0, clip_value_max=tf.cast(h, tf.float32) - 1)
        coords = tf.convert_to_tensor(coords, dtype=tf.float32)
        '''
        coords_0 = tf.clip_by_value(coords[..., 0], clip_value_min=0, clip_value_max=tf.cast(w, tf.float32) - 1)
        coords_1 = tf.clip_by_value(coords[..., 1], clip_value_min=0, clip_value_max=tf.cast(h, tf.float32) - 1)
        coords = tf.concat([tf.expand_dims(coords_0, axis=-1), tf.expand_dims(coords_1, axis=-1)], axis=-1)
        # 对一对坐标(x,y),转为四个整数floor(x),ceil(x),floor(y),ceil(y)
        # 得到四对坐标(floor(x),floor(y)),  (ceil(x),ceil(y)),
        # (floor(x),ceil(y)),  (ceil(x),floor(y))
        # tf.floor()向下取整,tf.ceil()向上取整，tf.stack()将values按照axis合并
        coords_lt = tf.cast(tf.math.floor(coords), tf.int32)  # (floor(x),floor(x))
        coords_rb = tf.cast(tf.math.ceil(coords), tf.int32)  # (ceil(x),ceil(y))
        coorbs_lb = tf.stack([coords_lt[..., 0], coords_rb[..., 1]], axis=-1)  # (floor(x),ceil(y))
        coords_rt = tf.stack([coords_rb[..., 0], coords_lt[..., 1]], axis=-1)  # (ceil(x),floor(y))

        # idx 为索引展开，形状为(b*c*h*w,) 表示全部图片和通道的坐标总和
        idx = self.tf_repeat(tf.range(bc), hw)

        def _get_vals_by_coords(input, coords):
            indices = tf.stack([idx, tf.reshape(coords[..., 0], shape=(-1)), tf.reshape(coords[..., 1], shape=(-1))],
                               axis=-1)
            vals = tf.gather_nd(input, indices)
            vals = tf.reshape(vals, shape=(bc, hw))
            return vals

        vals_lt = _get_vals_by_coords(inputs, coords_lt)
        vals_rb = _get_vals_by_coords(inputs, coords_rb)
        vals_lb = _get_vals_by_coords(inputs, coorbs_lb)
        vals_rt = _get_vals_by_coords(inputs, coords_rt)
        # 使用双线性插值得到像素值
        coords_offset_lt = coords - tf.cast(coords_lt, 'float32')
        vals_t = vals_lt + (vals_rt - vals_lt) * coords_offset_lt[..., 0]
        vals_b = vals_lb + (vals_rb - vals_lb) * coords_offset_lt[..., 0]
        mapped_vals = vals_t + (vals_b - vals_t) * coords_offset_lt[..., 1]
        return mapped_vals

    def to_b_h_w_c(self, x, x_shape):
        '''
        (b*c,h*w) -> (b,h,w,c)
        :param x:
        :param x_shape:
        :return:
        '''
        x = tf.reshape(x, shape=(-1, x_shape[3], x_shape[1], x_shape[2]))
        x = tf.transpose(x, perm=[0, 2, 3, 1])
        return x

    def call(self, inputs):
        '''返回形变后的feature map'''

        x_shape = tf.shape(inputs)  # (b,h,w,c)
        offsets = self.conv_offset(inputs)  # (b,h,w,2*c)
        offsets = self.to_bc_h_w_2(offsets, x_shape)  # (b*c,h,w,2)
        x = self.to_bc_h_w(inputs, x_shape)  # (b*c,h,w)
        # 通过offsets 和 inputs 映射生成坐标 (b*c,h*w,2)
        coords = self.get_coords(x, offsets)
        mapped_vals = self.tf_batch_map_coordinates(x, coords)  # (b*c,h*w)
        x_offset = self.to_b_h_w_c(mapped_vals, x_shape)
        return x_offset


if __name__ == '__main__':
    sources = tf.random.normal(shape=(4, 32, 32, 3))
    target = tf.constant([0, 1, 2, 3])
    network = models.Sequential([layers.Conv2D(filters=6, kernel_size=(3, 3), strides=(1, 1), padding='SAME'),
                                 ConvOffset2D(kernel_size=(3, 3), strides=(1, 1)),
                                 layers.GlobalAveragePooling2D(),
                                 layers.Dense(units=4),
                                 layers.Softmax()])
    loss_fn = losses.SparseCategoricalCrossentropy()
    optimizer = keras.optimizers.Adam(learning_rate=1e-3)
    with tf.GradientTape() as tape:
        predict = network(sources)
        loss = loss_fn(target, predict)
    gradients = tape.gradient(loss, network.trainable_variables)
    optimizer.apply_gradients(zip(gradients, network.trainable_variables))
    1