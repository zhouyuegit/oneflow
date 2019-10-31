#include "oneflow/core/kernel/kernel.h"
#include <opencv2/opencv.hpp>

namespace oneflow {

namespace {

template<typename T>
void ClearBlob(Blob* blob) {
  std::memset(blob->mut_dptr<T>(), 0, blob->shape().elem_cnt() * sizeof(T));
}

template<typename T>
T* GetPointerPtr(T* ptr, const int32_t height, const int32_t width, const int32_t x0,
                 const int32_t y0) {
  CHECK_LT(x0, width);
  CHECK_LT(y0, height);
  return ptr + y0 * width + x0;
}

template<typename T>
void CopyRegion(T* dst_ptr, const int32_t dst_step, const T* src_ptr, const int32_t src_step,
                const int32_t region_height, const int32_t region_width) {
  FOR_RANGE(int32_t, i, 0, region_height) {
    memcpy(dst_ptr + i * dst_step, src_ptr + i * src_step, region_width * sizeof(T));
  }
}

template<typename T>
void ExpandMaskProbAndBox(const int32_t mask_idx, const Blob* mask_prob_blob, const Blob* box_blob,
                          const int32_t padding, Blob* mask_prob_buf_blob, Blob* box_buf_blob) {
  // Expand Mask Prob
  const int32_t mask_h = mask_prob_blob->shape().At(1);
  const int32_t mask_w = mask_prob_blob->shape().At(2);
  const int32_t expanded_mask_h = mask_h + 2 * padding;
  const int32_t expanded_mask_w = mask_w + 2 * padding;
  CHECK_LE(expanded_mask_h, mask_prob_buf_blob->static_shape().At(0));
  CHECK_LE(expanded_mask_w, mask_prob_buf_blob->static_shape().At(1));
  mask_prob_buf_blob->dense_shape_mut_view().set_shape({expanded_mask_h, expanded_mask_w});
  ClearBlob<T>(mask_prob_buf_blob);
  CopyRegion(
      GetPointerPtr(mask_prob_buf_blob->mut_dptr<T>(), expanded_mask_h, expanded_mask_w, padding,
                    padding),
      expanded_mask_w,
      GetPointerPtr(mask_prob_blob->dptr<T>() + mask_idx * mask_h * mask_w, mask_h, mask_w, 0, 0),
      mask_w, mask_h, mask_w);

  // Expand Box
  const T* box_ptr = box_blob->dptr<T>() + mask_idx * 4;
  T* box_buf_ptr = box_buf_blob->mut_dptr<T>();
  const float scale_h = static_cast<float>(expanded_mask_h) / static_cast<float>(mask_h);
  const float scale_w = static_cast<float>(expanded_mask_w) / static_cast<float>(mask_w);
  const float h_half = (box_ptr[3] - box_ptr[1]) * 0.5 * scale_h;
  const float w_half = (box_ptr[2] - box_ptr[0]) * 0.5 * scale_w;
  const float x_c = (box_ptr[2] + box_ptr[0]) * 0.5;
  const float y_c = (box_ptr[3] + box_ptr[1]) * 0.5;
  box_buf_ptr[0] = x_c - w_half;
  box_buf_ptr[1] = y_c - h_half;
  box_buf_ptr[2] = x_c + w_half;
  box_buf_ptr[3] = y_c + h_half;
}

template<typename T>
int GetCVType();

template<>
int GetCVType<float>() {
  return CV_32FC1;
}
template<>
int GetCVType<double>() {
  return CV_64FC1;
}

template<typename T>
void GenBinaryMask(const int32_t mask_idx, Blob* mask_prob_buf_blob, const Blob* box_buf_blob,
                   const float threshold, Blob* mask_buf_blob) {
  const int32_t TO_REMOVE = 1;
  const T* box_buf_ptr = box_buf_blob->dptr<T>();
  const int32_t x0 = static_cast<int32_t>(box_buf_ptr[0]);
  const int32_t y0 = static_cast<int32_t>(box_buf_ptr[1]);
  const int32_t x1 = static_cast<int32_t>(box_buf_ptr[2]);
  const int32_t y1 = static_cast<int32_t>(box_buf_ptr[3]);
  const int32_t target_h = std::max(y1 - y0 + TO_REMOVE, 1);
  const int32_t target_w = std::max(x1 - x0 + TO_REMOVE, 1);
  cv::Mat origin_mat(mask_prob_buf_blob->shape().At(0), mask_prob_buf_blob->shape().At(1),
                     GetCVType<T>(), mask_prob_buf_blob->mut_dptr<T>());
  cv::Mat target_mat(target_h, target_w, GetCVType<T>());
  cv::resize(origin_mat, target_mat, cv::Size(target_h, target_w), 0, 0, cv::INTER_LINEAR);
  CHECK(target_mat.isContinuous());
  CHECK_LE(target_mat.cols, mask_buf_blob->static_shape().At(0));
  CHECK_LE(target_mat.rows, mask_buf_blob->static_shape().At(1));
  mask_buf_blob->dense_shape_mut_view().set_shape({target_mat.rows, target_mat.cols});
  ClearBlob<int8_t>(mask_buf_blob);
  FOR_RANGE(int32_t, i, 0, target_mat.rows * target_mat.cols) {
    mask_buf_blob->mut_dptr<int8_t>()[i] = static_cast<int8_t>(target_mat.at<T>(i) > threshold);
  }
}

template<typename T>
void PasteOnImage(const int32_t mask_idx, const Blob* mask_buf_blob, const Blob* box_buf_blob,
                  Blob* out_blob) {
  const int32_t img_h = out_blob->shape().At(1);
  const int32_t img_w = out_blob->shape().At(2);
  const T* box_buf_ptr = box_buf_blob->dptr<T>();
  const int32_t box_x0 = static_cast<int32_t>(box_buf_ptr[0]);
  const int32_t box_y0 = static_cast<int32_t>(box_buf_ptr[1]);
  const int32_t box_x1 = static_cast<int32_t>(box_buf_ptr[2]);
  const int32_t box_y1 = static_cast<int32_t>(box_buf_ptr[3]);
  const int32_t x0 = std::max(box_x0, 0);
  const int32_t y0 = std::max(box_y0, 0);
  const int32_t x1 = std::min(box_x1 + 1, img_w);
  const int32_t y1 = std::min(box_y1 + 1, img_h);

  const int32_t real_mask_h = y1 - y0 - 1;
  const int32_t real_mask_w = x1 - x0 - 1;

  CopyRegion(
      GetPointerPtr(out_blob->mut_dptr<int8_t>() + mask_idx * img_h * img_w, img_h, img_w, x0, y0),
      img_w,
      GetPointerPtr(mask_buf_blob->dptr<int8_t>(), real_mask_h, real_mask_w, x0 - box_x0,
                    y0 - box_y0),
      real_mask_w, real_mask_h, real_mask_w);
}

}  // namespace

template<typename T>
class MaskerKernel final : public KernelIf<DeviceType::kCPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MaskerKernel);
  MaskerKernel() = default;
  ~MaskerKernel() = default;

 private:
  void ForwardDenseShape(const KernelCtx& ctx,
                         std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    const Blob* mask_prob_blob = BnInOp2Blob("mask_prob");
    const Blob* image_size_blob = BnInOp2Blob("image_size");
    BnInOp2Blob("out")->dense_shape_mut_view().set_shape({mask_prob_blob->shape().At(0),
                                                          image_size_blob->dptr<int32_t>()[0],
                                                          image_size_blob->dptr<int32_t>()[1]});
  }
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    const MaskerOpConf& conf = this->op_conf().masker_conf();
    const Blob* mask_prob_blob = BnInOp2Blob("mask_prob");
    const Blob* box_blob = BnInOp2Blob("box");
    Blob* mask_prob_buf_blob = BnInOp2Blob("mask_prob_buf");
    Blob* box_buf_blob = BnInOp2Blob("box_buf");
    Blob* mask_buf_blob = BnInOp2Blob("mask_buf");
    Blob* out_blob = BnInOp2Blob("out");
    ClearBlob<int8_t>(out_blob);

    FOR_RANGE(int32_t, mask_idx, 0, out_blob->shape().At(0)) {
      ExpandMaskProbAndBox<T>(mask_idx, mask_prob_blob, box_blob, conf.padding(),
                              mask_prob_buf_blob, box_buf_blob);
      GenBinaryMask<T>(mask_idx, mask_prob_buf_blob, box_buf_blob, conf.threshold(), mask_buf_blob);
      PasteOnImage<T>(mask_idx, mask_buf_blob, box_buf_blob, out_blob);
    }
  }
};

REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kMaskerConf, DeviceType::kCPU, float,
                                      MaskerKernel<float>)
REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kMaskerConf, DeviceType::kCPU, double,
                                      MaskerKernel<double>)

}  // namespace oneflow
