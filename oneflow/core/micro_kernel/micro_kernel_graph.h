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
             const std::string& blob_name, const std::string& diff_blob_name)
      : mk_graph_(mk_graph),
        producer_mk_node_(producer_mk_node),
        blob_name_(blob_name),
        diff_blob_name_(diff_blob_name),
        used_cnt_(0) {}
  ~BlobSymbol() = default;

  //  Getters
  MicroKernelGraph* mut_mk_graph() { return mk_graph_; }
  MicroKernelNode* mut_producer_mk_node() { return producer_mk_node_; }
  bool NeedAccmulateDiff() const { return used_cnt_ > 1; }
  bool IsTrainable() const { return diff_blob_name_ != ""; }
  const std::string& blob_name() const { return blob_name_; }
  const std::string& diff_blob_name() const { return diff_blob_name_; }
  void WithDiffBlob(const std::function<Blob*(const std::string&)>& Blob4BnInOp,
                    const std::function<void(Blob*, bool is_acc)>& Handler) {
    if (IsTrainable()) {
      Blob* blob = Blob4BnInOp(diff_blob_name_);
      CHECK(blob);
      Handler(blob, NeedAccmulateDiff());
    }
  }

  // Setters
  void inc_used_cnt() { ++used_cnt_; }

 private:
  MicroKernelGraph* mk_graph_;
  MicroKernelNode* producer_mk_node_;
  std::string blob_name_;
  std::string diff_blob_name_;
  int32_t used_cnt_;
};

class BlobSymbolBuilder final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BlobSymbolBuilder);
  explicit BlobSymbolBuilder(MicroKernelGraph* mk_graph)
      : mk_graph_(mk_graph) {}
  ~BlobSymbolBuilder() = default;

  BlobSymbol* NewTmpBlobSymbol(const std::string& name);

  BlobSymbol* NewTrainableBlobSymbol(const std::string& name,
                                     const std::string& diff_name);

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
    InitAllBlobSymbolsUsedCnt();
  }
  ~MicroKernelGraph() = default;

  BlobSymbol* NewBlobSymbol(MicroKernelNode* producer_mk,
                            const std::string& name,
                            const std::string& diff_name) {
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
  void InitAllBlobSymbolsUsedCnt() {
    HashMap<std::string, HashSet<const BlobSymbol*>> blob_name2blob_symbols;
    for (const auto& blob_symbol : blob_symbols_) {
      const std::string& blob_name = blob_symbol->blob_name();
      blob_name2blob_symbols[blob_name].insert(blob_symbol.get());
      CHECK_EQ(blob_name2blob_symbols.at(blob_name).size(), 1);
      if (blob_symbol->IsTrainable()) {
        const std::string& diff_blob_name = blob_symbol->diff_blob_name();
        blob_name2blob_symbols[diff_blob_name].insert(blob_symbol.get());
        CHECK_EQ(blob_name2blob_symbols.at(diff_blob_name).size(), 1);
      }
    }
    for (const auto* mk_node : topo_mk_node_list_) {
      for (BlobSymbol* blob_symbol : mk_node->input_blob_symbols()) {
        blob_symbol->inc_used_cnt();
      }
    }
  }
  std::list<std::unique_ptr<BlobSymbol>> blob_symbols_;
  std::list<const MicroKernelNode*> topo_mk_node_list_;
};

inline BlobSymbol* BlobSymbolBuilder::NewTmpBlobSymbol(
    const std::string& name) {
  return mk_graph_->NewBlobSymbol(nullptr, name, "");
}

inline BlobSymbol* BlobSymbolBuilder::NewTrainableBlobSymbol(
    const std::string& name, const std::string& diff_name) {
  return mk_graph_->NewBlobSymbol(nullptr, name, diff_name);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_MICRO_KERNEL_MICRO_KERNEL_GRAPH_H_
