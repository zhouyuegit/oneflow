#include "oneflow/core/thread/cpu_thread.h"

namespace oneflow {

CpuThread::CpuThread(int64_t thrd_id, size_t buf_size) {
  set_thrd_id(thrd_id);
  mut_actor_thread() = std::thread([this, buf_size]() {
    void* buf_ptr = nullptr;
    if (buf_size > 0) {
      buf_ptr = malloc(buf_size);
      CHECK_NOTNULL(buf_ptr);
    }
    {
      ThreadCtx ctx;
      BlobDesc blob_desc =
          BlobDesc(Shape({static_cast<int64_t>(buf_size)}), DataType::kChar, false, false, 1);
      ctx.buf_blob.reset(
          NewBlob(nullptr, &blob_desc, static_cast<char*>(buf_ptr), nullptr, DeviceType::kCPU));
      ctx.cb_event_chan = nullptr;
      PollMsgChannel(ctx);
    }
    if (buf_ptr) { free(buf_ptr); }
  });
}

}  // namespace oneflow
