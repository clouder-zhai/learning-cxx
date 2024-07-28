#include "../exercise.h"

// READ: 类模板 <https://zh.cppreference.com/w/cpp/language/class_template>

template<class T>
struct Tensor4D {
    unsigned int shape[4];
    T *data;

    Tensor4D(unsigned int const shape_[4], T const *data_) {
        memcpy(shape, shape_, 4 * sizeof(unsigned int));
        unsigned int size = shape[0] * shape[1] * shape[2] * shape[3];
        // TODO: 填入正确的 shape 并计算 size
        data = new T[size];
        std::memcpy(data, data_, size * sizeof(T));
    }
    ~Tensor4D() {
        delete[] data;
    }

    // 为了保持简单，禁止复制和移动
    Tensor4D(Tensor4D const &) = delete;
    Tensor4D(Tensor4D &&) noexcept = delete;

    // 这个加法需要支持“单向广播”。
    // 具体来说，`others` 可以具有与 `this` 不同的形状，形状不同的维度长度必须为 1。
    // `others` 长度为 1 但 `this` 长度不为 1 的维度将发生广播计算。
    // 例如，`this` 形状为 `[1, 2, 3, 4]`，`others` 形状为 `[1, 2, 1, 4]`，
    // 则 `this` 与 `others` 相加时，3 个形状为 `[1, 2, 1, 4]` 的子张量各自与 `others` 对应项相加。
    Tensor4D &operator+=(Tensor4D const &others) {
        // TODO: 实现单向广播的加法
        if(this->shape[0] != others.shape[0]) {
            ASSERT(false, "shape not match");
            return *this;
        }
        if((shape[1] != others.shape[1] && others.shape[1] != 1) || (shape[2] != others.shape[2] && others.shape[2] != 1) || (shape[3] != others.shape[3] && others.shape[3] != 1)) {
            ASSERT(false, "cannot propagate.");
            return *this;
        }
        bool flag3 = false;
        if(shape[1] != others.shape[1]) flag3 = true;
        bool flag1 = false;
        if(shape[2] != others.shape[2]) flag1 = true;
        bool flag2 = false;
        if(shape[3] != others.shape[3]) flag2 = true;
        for(int i = 0; i < this->shape[0]; ++i) {
            int offset_1_data = shape[1] * shape[2] * shape[3];
            int offset_1_other = others.shape[1] * others.shape[2] * others.shape[3];
            for(int j = 0; j < shape[1]; ++j) {
                int offset_2_data = shape[2] * shape[3];
                int offset_2_other = others.shape[2] * others.shape[3];
                for(int m = 0; m < shape[2]; ++m) {
                    int offset_3_data = shape[3];
                    int offset_3_other = others.shape[3];
                    for(int n = 0; n < shape[3]; ++n) {
                        int idx_data = offset_1_data * i + offset_2_data * j + offset_3_data * m + n;
                        int idx_other = offset_1_other * i;
                        if(!flag3) {
                            idx_other += offset_2_other * j;
                        }
                        if(!flag1) {
                            idx_other += offset_3_other * m;
                        }
                        if(!flag2) {
                            idx_other += n;
                        }
                        data[idx_data] += others.data[idx_other];
                    }
                }
            }
        }
        return *this;
    }
};

// ---- 不要修改以下代码 ----
int main(int argc, char **argv) {
    {
        unsigned int shape[]{1, 2, 3, 4};
        // clang-format off
        int data[]{
             1,  2,  3,  4,
             5,  6,  7,  8,
             9, 10, 11, 12,

            13, 14, 15, 16,
            17, 18, 19, 20,
            21, 22, 23, 24};
        // clang-format on
        auto t0 = Tensor4D<int>(shape, data);
        auto t1 = Tensor4D<int>(shape, data);
        t0 += t1;
        for (auto i = 0u; i < sizeof(data) / sizeof(*data); ++i) {
            ASSERT(t0.data[i] == data[i] * 2, "Tensor doubled by plus its self.");
        }
    }
    {
        unsigned int s0[]{1, 2, 3, 4};
        // clang-format off
        float d0[]{
            1, 1, 1, 1,
            2, 2, 2, 2,
            3, 3, 3, 3,

            4, 4, 4, 4,
            5, 5, 5, 5,
            6, 6, 6, 6};
        // clang-format on
        unsigned int s1[]{1, 2, 3, 1};
        // clang-format off
        float d1[]{
            6,
            5,
            4,

            3,
            2,
            1};
        // clang-format on

        auto t0 = Tensor4D<float>(s0, d0);
        auto t1 = Tensor4D<float>(s1, d1);
        t0 += t1;
        for (auto i = 0u; i < sizeof(d0) / sizeof(*d0); ++i) {
            ASSERT(t0.data[i] == 7.f, "Every element of t0 should be 7 after adding t1 to it.");
        }
    }
    {
        unsigned int s0[]{1, 2, 3, 4};
        // clang-format off
        double d0[]{
             1,  2,  3,  4,
             5,  6,  7,  8,
             9, 10, 11, 12,

            13, 14, 15, 16,
            17, 18, 19, 20,
            21, 22, 23, 24};
        // clang-format on
        unsigned int s1[]{1, 1, 1, 1};
        double d1[]{1};

        auto t0 = Tensor4D<double>(s0, d0);
        auto t1 = Tensor4D<double>(s1, d1);
        t0 += t1;
        for (auto i = 0u; i < sizeof(d0) / sizeof(*d0); ++i) {
            ASSERT(t0.data[i] == d0[i] + 1, "Every element of t0 should be incremented by 1 after adding t1 to it.");
        }
    }
}
