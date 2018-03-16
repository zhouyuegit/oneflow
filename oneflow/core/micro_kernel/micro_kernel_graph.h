#ifndef ONEFLOW_CORE_MICRO_KERNEL_MICRO_KERNEL_GRAPH_H_
#define ONEFLOW_CORE_MICRO_KERNEL_MICRO_KERNEL_GRAPH_H_

#include "oneflow/core/graph/graph.h"
#include "oneflow/core/register/blob_desc.h"
#include "oneflow/core/job/resource.pb.h"

namespace oneflow {

class MicroKernelNode;

class MicroKernelEdge final : public Edge<MicroKernelNode, MicroKernelEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MicroKernelEdge);
  MicroKernelEdge() = default;
  ~MicroKernelEdge() = default;
};

class MicroKernelGraph;

class MicroKernelNode : public Node<MicroKernelNode, MicroKernelEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MicroKernelNode);
  MicroKernelNode() = default;
  virtual ~MicroKernelNode() = default;
};

class BlobSymbol final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BlobSymbol);
  BlobSymbol(MicroKernelGraph* mk_graph_,
	     MicroKernelNode* producer_mk_node,
	     const std::string& blob_name, const std::string& diff_blob_name)
    : mk_graph(mk_graph), producer_mk_node_(producer_mk_node),
      blob_name_(blob_name), diff_blob_name_(diff_blob_name) {}
  ~BlobSymbol() = default;

  //  Getters
  MicroKernelGraph* mut_mk_graph() { return mk_graph_; }
  MicroKernelNode* mut_producer_mk_node() { return producer_mk_node_; }
  Bool IsTrainable() const { return diff_blob_name_ != ""; }
  const std::string& blob_name() const { return blob_name_; }
  void WithDiffBlob(const std::function<Blob*(const std::string&)>& Blob4BnInOp,
		    const std::function<void(Blob*)>& Handler) {
    TODO();
  }
  
 private:
  MicroKernelGraph* mk_graph_;
  MicroKernelNode* producer_mk_node_;
  std::string blob_name_;
  std::string diff_blob_name_;
};

template<DeviceType device_type, typename T> class MicroKernel;

class MicroKernelGraph final : public Graph<MicroKernelNode, MicroKernelEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MicroKernelGraph);
  MicroKernelGraph() = default;
  ~MicroKernelGraph() = default;

  BlobSymbol* NewBlobSymbol(MicroKernelNode* producer_mk,
			    const std::string& name, const std::string& diff_name) {
    TODO();
  }

  BlobSymbol* NewTmpBlobSymbol(const std::string& name) {
    return NewBlobSymbol(nullptr, name, "");
  }
  
  BlobSymbol* NewTrainableBlobSymbol(const std::string& name,
				     const std::string& diff_name) {
    return NewBlobSymbol(nullptr, name, diff_name);
  }

  template<DeviceType device_type, typename T>
  void Forward(const KernelCtx& device_ctx,
	       const std::function<Blob*(const std::string&)>& Blob4BnInOp) const  {
    TopoForEachNode([&](MicroKernelNode* mk_node){
      auto* micro_kernel = dynamic_cast<MicroKernel<device_type, T>*>(mk_node);
      CHECK(micro_kernel);
      micro_kernel->Forward(device_ctx, Blob4BnInOp);
    });
  }
 
  template<DeviceType device_type, typename T>
  void Backward(const KernelCtx& device_ctx,
	       const std::function<Blob*(const std::string&)>& Blob4BnInOp) const {
    ReverseTopoForEachNode([&](MicroKernelNode* mk_node){
      auto* micro_kernel = dynamic_cast<MicroKernel<device_type, T>*>(mk_node);
      CHECK(micro_kernel);
      micro_kernel->Backward(device_ctx, Blob4BnInOp);
    });
  }

 private:
  std::list<std::unique_ptr<BlobSymbol>> blob_symbols_;
};

}

#endif  // ONEFLOW_CORE_MICRO_KERNEL_MICRO_KERNEL_GRAPH_H_
