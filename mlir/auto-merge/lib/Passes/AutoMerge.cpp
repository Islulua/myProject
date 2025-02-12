#include "MyProject/Passes/AutoMerge.h"
#include "mlir/Pass/Pass.h"

namespace tx8be_mlir {
namespace auto_merge {

class GroupRegion {
public:
  GroupRegion(Operation *op) : ops_{op} {}

  void add(std::shared_ptr<GroupRegion> region) {
    ops_.insert(ops_.end(), region->getOps().begin(), region->getOps().end());
  }
  llvm::SmallVector<Operation *> getOps() {
    return ops_;
  }

  void clear() {
    ops_.clear();
  }

  size_t getCycle();

  llvm::SmallVector<std::shared_ptr<GroupRegion>> getUserRegions();
  static llvm::MapVector<Operation *, std::shared_ptr<GroupRegion>> opRegionMap;
  static llvm::MapVector<Operation *, std::set<Operation *>> opDefsMap;
  static void initDefsMap(tx8be::SubGraphOp &subGraphOp);

private:
  std::vector<Operation *> ops_;
};

static void initDefsMap(tx8be::SubGraphOp &subGraphOp) {
  subGraphOp.walk([&](Operation *op) {
    for (auto operand : op->getOperands()) {
      auto def = operand.getDefiningOp();
      if (!def) continue;
      opDefsMap[op].insert(def);
      opDefsMap[op].insert(opDefsMap[def].begin(), opDefsMap[def].end());
    }
  });
}

llvm::SmallVector<std::shared_ptr<GroupRegion>> GroupRegion::getUserRegions() {
  llvm::SmallVector<std::shared_ptr<GroupRegion>> userRegions;
  std::unordered_set<Operation *> opsSet(ops_.begin(), ops_.end());

  for (auto op : ops_) {
    for (auto user : op->getUsers()) {
      if (opsSet.find(user) == opsSet.end()) {
        auto it = opRegionMap.find(user);
        if (it != opRegionMap.end()) {
          userRegions.push_back(it->second);
        }
      }
    }
  }
  return userRegions;
}

tx8be::GroupOp createOpsGroupOp(tx8be::SubGraphOp &subGraphOp, const llvm::SmallVector<Operation *> &ops) {
  llvm::SmallVector<Operation *> dependencies;
  llvm::SmallVector<Value> groupInputs, groupOutputs;
  llvm::SmallVector<Type> groupInputsTypes, groupOutputsTypes;

  //记录group的输入输出
  for (auto op : ops) {
    for (auto operand : op->getOperands()) {
      auto def = operand.getDefiningOp();
      if (defOp && (isa<tx8be::NoneOp, tx8be::LoadConstOp, tx8be::ConstantOp>(defOp))) {
        dependencies.push_back(defOp);
      } else {
        if (std::find(ops.begin(), ops.end(), def) == ops.end() &&
           (std::find(groupInputs.begin(), groupInputs.end(), operand) == groupInputs.end())) {
          groupInputs.push_back(operand);
          groupInputsTypes.push_back(operand.getType());
        }
      }
    }
    for (auto user : op->getUsers()) {
      for (auto result : user->getResults()) {
        if ((std::find(groupOutputs.begin(), groupOutputs.end(), result) == groupOutputs.end()) &&
            (std::find(ops.begin(), ops.end(), user) == ops.end())) {
          groupOutputs.push_back(result);
          groupOutputsTypes.push_back(result.getType());
        }
      }
    }
  }

  OpBuilder builder(subGraphOp.getContext());
  // TODO
  // 创建groupOp
  auto groupOp = builder.create<tx8be::GroupOp>(subGraphOp.getLoc(), groupInputs, groupInputsTypes, groupOutputs, groupOutputsTypes, dependencies);
  return groupOp;
}

size_t GroupRegion::getCycle() {
  //TODO: 基于Key，将Key-Cycle的映射记录下来，后续如果遇到相同的模式直接返回Cycle值
  size_t key = getGroupKey(ops_);
  auto it = groupKeyCycleMap.find(key);
  if (it != groupKeyCycleMap.end()) {
    return it->second;
  }

  // 克隆所有的ops
  llvm::SmallVector<Operation *> clonedOps;
  for (auto op : ops_) {
    clonedOps.push_back(op->clone());
  }
  // 基于克隆的算子构造出一个克隆的groupOp
  auto groupOp = createOpsGroupOp(subGraphOp, clonedOps);
  
  if (failed(runLayerGroupPasses(groupOp))) {
    return CYCLE_MAX;
  }
  auto cycle = getGroupCycle(groupOp);
  groupOp.erase();
  return cycle;
}

class MergeStrategy {
public:
  virtual ~MergeStrategy() = default;
  virtual size_t evaluateBenefit(std::shared_ptr<GroupRegion> region, std::shared_ptr<GroupRegion> userRegion) = 0;
};

class DefaultMergeStrategy : public MergeStrategy {
public:
  size_t evaluateBenefit(std::shared_ptr<GroupRegion> region, std::shared_ptr<GroupRegion> userRegion) override {
    // 计算当前region的cycle
    size_t currentCycle = region->getCycle();
    // 计算userRegion的cycle
    size_t userCycle = userRegion->getCycle();
    // 创建一个临时region，用于计算合并后的cycle
    auto mergedRegion = std::make_shared<GroupRegion>(*region);
    mergedRegion->add(userRegion);
    size_t mergedCycle = mergedRegion->getCycle();
    return mergedCycle - currentCycle - userCycle;
  }
};

class TopologicalSortStrategy {
public:
  virtual ~TopologicalSortStrategy() = default;
  virtual bool canSort(llvm::SmallVector<Operation *> ops) = 0;
};

bool isOpReady(Operation *op, DenseSet<Operation *> &unscheduledOps) {
  // 当满意以下条件的时候，算子被认为是ready的：
  const auto isReady = [&](Value value) {
    Operation *parent = value.getDefiningOp();
    // --- 如果输入来源于参数，则认为输入ready
    if (!parent) return true;
    // --- 如果这个value不是被unschduledOps中的算子定义，则认为该value是ready的
    do {
      if (parent == op) return true;
      if (unscheduledOps.contains(parent)) return false;
    } while (parent = parent.getParentOp());
    return true;
  }
  WalkResult readyToSchedule = op->walk([&](Operation *nestedOp) {
    return llvm::all_of(nestedOp->getOperands(), [&](Value operand) { return isReady(value); })
                        ? WalkResult::advance()
                        : WalkResult::interrupt();
  });
  return !readyToSchedule.wasInterrupted();
}

class DefaultTopologicalSortStrategy : public TopologicalSortStrategy {
public:
  bool canSort(llvm::SmallVector<Operation *> ops) override {
    if (ops.size() <= 1) return true;
    DenseSet<Operation *> unscheduledOps;
    for (auto op : ops) {
      unscheduledOps.insert(op);
    }
    auto nextScheduledOp = ops.begin();
    auto end             = ops.end();
  
    bool allOpsScheduled = true;
    while (!unscheduledOps.empty()) {
      bool scheduledAtLeastOnce = false;
      for (Operation &op : llvm::make_early_inc_range(llvm::make_range(nextScheduledOp, end))) {
        if (!isOpReady(op, unscheduledOps)) continue;
        unscheduledOps.erase(&op);
        scheduledAtLeastOnce = true;
        if (&op == &*nextScheduledOp) {
          ++nextScheduledOp;
        }
      }
      if (!scheduledAtLeastOnce) {
        allOpsScheduled = false;
        unscheduledOps.erase(&*nextScheduledOp);
        ++nextScheduledOp;
      }
    }
    return allOpsScheduled;
  }
};

struct AutoMergePass : public PassWrapper<AutoMergePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AutoMergePass)

  void runOnOperation() override;
};  

void AutoMergePass::runOnOperation() {
  auto funcOp = getOperation();
  llvm::SmallVector<Operation *> order;
  llvm::SmallVector<std::shared_ptr<GroupRegion>> regions;

  // 遍历funcOp，初始化opRegionMap和regions
  // opRegionMap 记录op和GroupRegion的映射
  // regions 记录所有的GroupRegion
  funcOp->walk([&](Operation *op) {
    if (isa<tx8be::NoneOp, tx8be::LoadConstOp>(op)) {
      return;
    }
    order.push_back(op);
    GroupRegion::opRegionMap[op] = std::make_shared<GroupRegion>(op);
    regions.push_back(GroupRegion::opRegionMap[op]);
  });

  // 创建合并策略和排序策略，使用策略模式，方便后续扩展
  auto mergeStrategy = std::make_unique<DefaultMergeStrategy>();
  auto sortStrategy = std::make_unique<DefaultTopologicalSortStrategy>();

  // 进行合并，直到没有合并发生为止
  do {
    bool merged = false; // 标志变量，跟踪是否发生合并
    // 遍历所有region，尝试合并, 每个region与自己的user region尝试进行合并
    for (auto region : regions) {
      for (auto userRegion : region->getUserRegions()) {
        // 创建一个临时region，用于检查是否可以排序，基于图的拓扑结构
        auto tempRegion = std::make_shared<GroupRegion>(*region);
        tempRegion->add(userRegion);
        if (!sortStrategy->canSort(tempRegion->getOps())) continue;

        // 如果可以进行合并，计算合并的收益, 如果收益大于0，则进行合并
        int benefit = mergeStrategy->evaluateBenefit(region, userRegion);
        if (benefit > 0) {
          region->add(userRegion);
          userRegion->clear();
          merged = true; // 如果发生合并，设置标志为 true
        }
      }
    }
  } while (merged); // 如果没有发生合并，退出循环

  llvm::SmallVector<Operation *> groupOps;
  for (auto region : regions) {
    if (!region->empty()) {
      auto groupOp = createAndInsertGroupOp(region->getOps());
      groupOps.push_back(groupOp);
    }
  }
  for (auto groupOp : groupOps) {
    sortTopologically(groupOp);
  }
}

// 假设有一个函数来创建和插入 GroupOp
void createAndInsertGroupOp(tx8be::SubGraphOp &subGraphOp, const llvm::SmallVector<Operation *> &ops) {

}

std::unique_ptr<Pass> createAutoMergePass() {
  return std::make_unique<AutoMergePass>();
}

} // namespace myproject
} // namespace mlir