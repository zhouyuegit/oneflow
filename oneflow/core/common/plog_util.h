#ifndef ONEFLOW_CORE_COMMON_PLOG_UTIL_H
#define ONEFLOW_CORE_COMMON_PLOG_UTIL_H

#include <plog/Log.h>
#include <plog/Util.h>
#include <plog/Record.h>

namespace oneflow {
  using namespace plog;
  class JsonFmt {
    public:
      static util::nstring header() {
        return util::nstring();
      }

      static util::nstring format(const Record& record) {
        util::nostringstream ss;

        ss << record.getMessage() << "\n";
        return ss.str();
      }
  };
}
#endif

