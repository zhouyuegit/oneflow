#ifndef ONEFLOW_CORE_MICRO_KERNEL_MICRO_KERNEL_GRAPH_H_
#define ONEFLOW_CORE_MICRO_KERNEL_MICRO_KERNEL_GRAPH_H_

#include "oneflow/core/graph/graph.h"
#include "oneflow/core/register/blob.h"
#include "oneflow/core/job/resource.pb.h"
#include "oneflow/core/kernel/kernel_context.h"

namespace oneflow {

class MicroKernelNode;

class MicroKernelEdge final : public Edge<MicroKernelNode, MicroKernelEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MicroKernelEdge);
  MicroKernelEdge() = default;
  ~MicroKernelEdge() = default;
};

class MicroKernelGraph;

class BlobSymbol;

class MicroKernelNode : public Node<MicroKernelNode, MicroKernelEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MicroKernelNode);
  explicit MicroKernelNode(const std::vector<BlobSymbol*>& input_blob_symbols)
      : input_blob_symbols_(input_blob_symbols) {}
  virtual ~MicroKernelNode() = default;

  const std::vector<BlobSymbol*>& input_blob_symbols() const {
    return input_blob_symbols_;
  }

 private:
  std::vector<BlobSymbol*> input_blob_symbols_;
};

class BlobSymbol final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BlobSymbol);
  BlobSymbol(MicroKernelGraph* mk_graph, MicroKernelNode* producer_mk_node,
             const std::string& blob_name, const std::string& diff_blob_name,
             bool need_accumulated_diff)
      : mk_graph_(mk_graph),
        producer_mk_node_(producer_mk_node),
        blob_name_(blob_name),
        diff_blob_name_(diff_blob_name),
        need_accumulated_diff_(need_accumulated_diff) {}
  ~BlobSymbol() = default;

  //  Getters
  MicroKernelGraph* mut_mk_graph() { return mk_graph_; }
  MicroKernelNode* mut_producer_mk_node() { return producer_mk_node_; }
  bool need_accumulated_diff() const { return need_accumulated_diff_; }
  bool IsTrainable() const { return diff_blob_name_ != ""; }
  const std::string& blob_name() const { return blob_name_; }
  const std::string& diff_blob_name() const { return diff_blob_name_; }
  void WithDiffBlob(const std::function<Blob*(const std::string&)>& Blob4BnInOp,
                    const std::function<void(Blob*, bool is_acc)>& Handler) {
    if (IsTrainable()) {
      Blob* blob = Blob4BnInOp(diff_blob_name_);
      CHECK(blob);
      Handler(blob, need_accumulated_diff());
    }
  }

  //  Setters
  void set_producer_mk_node(MicroKernelNode* producer_mk_node) {
    producer_mk_node_ = producer_mk_node;
  }

 private:
  MicroKernelGraph* mk_graph_;
  MicroKernelNode* producer_mk_node_;
  std::string blob_name_;
  std::string diff_blob_name_;
  bool need_accumulated_diff_;
};

class BlobSymbolBuilder final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BlobSymbolBuilder);
  explicit BlobSymbolBuilder(MicroKernelGraph* mk_graph)
      : mk_graph_(mk_graph) {}
  ~BlobSymbolBuilder() = default;

  BlobSymbol* NewTmpBlobSymbol(const std::string& name);

  BlobSymbol* NewTrainableBlobSymbol(const std::string& name,
                                     const std::string& diff_name,
                                     bool need_accumulated_diff);

 private:
  MicroKernelGraph* mk_graph_;
};

template<DeviceType device_type, typename T>
class MicroKernel;

class MicroKernelGraph final : public Graph<MicroKernelNode, MicroKernelEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MicroKernelGraph);
  explicit MicroKernelGraph(
      const std::function<void(BlobSymbolBuilder*)>& Build) {
    BlobSymbolBuilder blob_symbol_builer(this);
    Build(&blob_symbol_builer);
    InitTopoMicroKernelList();
  }
  ~MicroKernelGraph() = default;

  BlobSymbol* NewBlobSymbol(MicroKernelNode* producer_mk,
                            const std::string& name,
                            const std::string& diff_name,
                            bool need_accumulated_diff) {
    TODO();
  }

  template<DeviceType device_type, typename T>
  void Forward(
      const KernelCtx& device_ctx,
      const std::function<Blob*(const std::string&)>& Blob4BnInOp) const {
    for (const auto* mk_node : topo_mk_node_list_) {
      const auto* micro_kernel =
          dynamic_cast<MicroKernel<device_type, T>*>(mk_node);
      CHECK(micro_kernel);
      micro_kernel->Forward(device_ctx, Blob4BnInOp);
    }
  }

  template<DeviceType device_type, typename T>
  void Backward(
      const KernelCtx& device_ctx,
      const std::function<Blob*(const std::string&)>& Blob4BnInOp) const {
    for (auto micro_kernel_it = topo_mk_node_list_.rbegin();
         micro_kernel_it != topo_mk_node_list_.rend(); ++micro_kernel_it) {
      const auto* micro_kernel =
          dynamic_cast<MicroKernel<device_type, T>*>(*micro_kernel_it);
      CHECK(micro_kernel);
      micro_kernel->Backward(device_ctx, Blob4BnInOp);
    }
  }

 private:
  void InitTopoMicroKernelList() {
    TopoForEachNode([&](MicroKernelNode* mk_node) {
      topo_mk_node_list_.push_back(mk_node);
    });
  }
  std::list<std::unique_ptr<BlobSymbol>> blob_symbols_;
  std::list<const MicroKernelNode*> topo_mk_node_list_;
};

inline BlobSymbol* BlobSymbolBuilder::NewTmpBlobSymbol(
    const std::string& name) {
  return mk_graph_->NewBlobSymbol(nullptr, name, "", false);
}

inline BlobSymbol* BlobSymbolBuilder::NewTrainableBlobSymbol(
    const std::string& name, const std::string& diff_name,
    bool need_accumulated_diff) {
  return mk_graph_->NewBlobSymbol(nullptr, name, diff_name,
                                  need_accumulated_diff);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_MICRO_KERNEL_MICRO_KERNEL_GRAPH_H_
