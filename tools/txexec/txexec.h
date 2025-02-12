#ifndef TXEXEC_H
#define TXEXEC_H

#include "llvm/Support/CommandLine.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include <string>
#include <vector>



namespace my_project {
namespace tools {

class TxExecContext {
public:
    TxExecContext(std::string modulePath);
    ~TxExecContext();

    // Initialize Context
    bool initialize();

    // Execute the main functionality of the tool
    bool execute();

    // Clean up resources
    void cleanup();

private:
    // Load module from filePath
    void loadModule();

private:
    std::string modulePath_;
    mlir::OwningOpRef<mlir::ModuleOp> module_;
};
    
} // namespace tools
} // namespace my_project
#endif // TXEXEC_H