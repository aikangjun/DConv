from custom import *
import tensorflow as tf
import numpy as np
import tensorflow as tf


class ConvOffset2D(layers.Layer):
    """ConvOffset2D"""

    def __init__(self,
                 kernel_size: tuple,
                 strides: tuple,
                 **kwargs):
        """Init"""
        super(ConvOffset2D, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.strides = strides

    def build(self, input_shape):
        self.conv = layers.Conv2D(filters=input_shape[-1] * 2,
                                  kernel_size=self.kernel_size,
                                  strides=self.strides,
                                  padding='SAME')
        self.built = True

    def tf_flatten(self, a):
        """Flatten tensor"""
        return tf.reshape(a, [-1])

    def tf_repeat(self, a, repeats, axis=0):
        """TensorFlow version of np.repeat for 1D"""
        assert len(a.get_shape()) == 1

        a = tf.expand_dims(a, -1)
        a = tf.tile(a, [1, repeats])
        a = self.tf_flatten(a)
        return a

    def tf_repeat_2d(self, a, repeats):
        """Tensorflow version of np.repeat for 2D"""

        assert len(a.get_shape()) == 2  # 二维
        a = tf.expand_dims(a, 0)  # 在第0维之前扩一维
        a = tf.tile(a, [repeats, 1, 1])  # 在第0维重复repeats次
        return a

    def tf_batch_map_coordinates(self, input, coords, order=1):
        """Batch version of tf_map_coordinates
        Only supports 2D feature maps
        Parameters
        ----------
        input : tf.Tensor. shape = (b, s, s)
        coords : tf.Tensor. shape = (b, n_points, 2)
        """

        input_shape = tf.shape(input)
        bc = input_shape[0]
        h = input_shape[1]
        w = input_shape[2]
        n_coords = tf.shape(coords)[1]

        # coords = tf.clip_by_value(coords, 0,
        #                           tf.cast(h, 'float32') - 1)  # 基于定义的min与max对tesor数据进行截断操作，目的是为了应对梯度爆发或者梯度消失的情况

        coords_0 = tf.clip_by_value(coords[..., 0], clip_value_min=0, clip_value_max=tf.cast(w, tf.float32) - 1)
        coords_1 = tf.clip_by_value(coords[..., 1], clip_value_min=0, clip_value_max=tf.cast(h, tf.float32) - 1)
        coords = tf.concat([tf.expand_dims(coords_0,axis=-1), tf.expand_dims(coords_1,axis=-1)], axis=-1)

        coords_lt = tf.cast(tf.floor(coords), 'int32')  # 双线性插值，左上角的值，所有坐标向下取整
        coords_rb = tf.cast(tf.math.ceil(coords), 'int32')  # 右下角的值，向上取整
        coords_lb = tf.stack([coords_lt[..., 0], coords_rb[..., 1]], axis=-1)  # 左下角的值是，x最小，y最大，按通道堆叠左上角的x，右下角的y即可
        coords_rt = tf.stack([coords_rb[..., 0], coords_lt[..., 1]], axis=-1)  # 同上，x最大y嘴小

        idx = self.tf_repeat(tf.range(bc), n_coords)

        def _get_vals_by_coords(input, coords):
            indices = tf.stack([
                idx, self.tf_flatten(coords[..., 0]), self.tf_flatten(coords[..., 1])
            ], axis=-1)  # 根据batch,x,y建立索引
            vals = tf.gather_nd(input, indices)  # 取得输入对应索引位置处的值,vals为一维
            vals = tf.reshape(vals, (bc, n_coords))  # 转化成二维
            return vals

        vals_lt = _get_vals_by_coords(input, coords_lt)  # 获取四个角的像素值
        vals_rb = _get_vals_by_coords(input, coords_rb)
        vals_lb = _get_vals_by_coords(input, coords_lb)
        vals_rt = _get_vals_by_coords(input, coords_rt)

        coords_offset_lt = coords - tf.cast(coords_lt, 'float32')  # 进行双线性插值，得到目标坐标的像素值
        vals_t = vals_lt + (vals_rt - vals_lt) * coords_offset_lt[..., 0]
        vals_b = vals_lb + (vals_rb - vals_lb) * coords_offset_lt[..., 0]
        mapped_vals = vals_t + (vals_b - vals_t) * coords_offset_lt[..., 1]

        return mapped_vals  # 得到偏移后坐标的所有像素值

    def tf_batch_map_offsets(self, input, offsets, order=1):
        """Batch map offsets into input
        Parameters
        ---------
        input : tf.Tensor. shape = (b, s, s)
        offsets: tf.Tensor. shape = (b, s, s, 2)
        """

        input_shape = tf.shape(input)  # (bc,h,w)
        bc = input_shape[0]  # bc
        h = input_shape[1]
        w = input_shape[2]

        offsets = tf.reshape(offsets, (bc, -1, 2))  # (bc,h*w,2)
        grid = tf.meshgrid(
            tf.range(h), tf.range(w), indexing='ij'
        )  # 广播，将一个以为tensor进行广播，当存在两个输入时=（a,b）,
        # 先将a按行广播为size(b)行，再将b按列广播为size(a)列(当index=‘xy’时，为笛卡尔坐标系，当index=‘ij’,则为矩阵坐标系，将前面顺序交换)
        grid = tf.stack(grid, axis=-1)  # 将两个通道堆叠在一起，则生成一个2通道的tensor,shape=(h,w,2)/(3,3,2),每个位置上是一个坐标(ij)
        grid = tf.cast(grid, 'float32')
        grid = tf.reshape(grid, (-1, 2))  # (h*w,2) 变成二维，每个元素表示一个坐标
        grid = self.tf_repeat_2d(grid, bc)  # 重复第0维,bc次，shape=(bc,h*h,2)
        coords = offsets + grid  # 每个通道的坐标都加上偏移量
        # 坐标变成了小数，需要
        mapped_vals = self.tf_batch_map_coordinates(input, coords)
        return mapped_vals

    @staticmethod
    def _to_bc_h_w_2(x, x_shape):
        """(b, h, w, 2c) -> (b*c, h, w, 2)"""
        x = tf.transpose(x, [0, 3, 1, 2])  # 交换维度(b,2c,h,w)
        x = tf.reshape(x, (-1, int(x_shape[1]), int(x_shape[2]), 2))  # (bc,h,w,2)
        return x

    @staticmethod
    def _to_bc_h_w(x, x_shape):
        """(b, h, w, c) -> (b*c, h, w)"""
        x = tf.transpose(x, [0, 3, 1, 2])  # 交换维度(b,c,h,w)
        x = tf.reshape(x, (-1, int(x_shape[1]), int(x_shape[2])))  # (bc,h,w)
        return x

    @staticmethod
    def _to_b_h_w_c(x, x_shape):
        """(b*c, h*w) -> (b, h, w, c)"""
        x = tf.reshape(
            x, (-1, int(x_shape[3]), int(x_shape[1]), int(x_shape[2]))
        )
        x = tf.transpose(x, [0, 2, 3, 1])
        return x

    def call(self, x):
        # TODO offsets probably have no nonlinearity?
        x_shape = x.get_shape()  # 输入tensor的shape=(b,h,w,c)
        offsets = self.conv(x)  # 进行对输入卷积，2*channels,shape=(b,h,w,2c)

        offsets = self._to_bc_h_w_2(offsets, x_shape)  # 将offses的shape转化为(bc,h,w,2),两个通道分别表示x,y的偏移量
        x = self._to_bc_h_w(x, x_shape)  # 将输入shape变为(bc,h,w)
        x_offset = self.tf_batch_map_offsets(x, offsets)  # 得到片以后新坐标的所有像素值
        x_offset = self._to_b_h_w_c(x_offset, x_shape)  # 变换维度
        return x_offset

    def compute_output_shape(self, input_shape):
        # 输出形状与输出形状相同
        return input_shape


if __name__ == '__main__':
    source = tf.random.normal(shape=(2, 10, 4, 3))
    target = tf.constant([0, 1])
    network = models.Sequential([layers.Conv2D(filters=3, kernel_size=(3, 3), strides=(1, 1), padding='SAME'),
                                 ConvOffset2D(kernel_size=(3, 3), strides=(1, 1)),
                                 layers.GlobalAveragePooling2D(),
                                 layers.Dense(units=2),
                                 layers.Softmax()])
    loss_fn = losses.SparseCategoricalCrossentropy()
    optimizer = keras.optimizers.Adam(learning_rate=1e-3)
    with tf.GradientTape() as tape:
        predict = network(source)
        loss = loss_fn(target, predict)
    gradients = tape.gradient(loss, network.trainable_variables)
    optimizer.apply_gradients(zip(gradients, network.trainable_variables))
    1
