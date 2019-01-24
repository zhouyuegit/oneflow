#ifndef ONEFLOW_CORE_COMMON_PLOG_UTIL_H
#define ONEFLOW_CORE_COMMON_PLOG_UTIL_H

#define PLOG_CAPTURE_FILE

#include <plog/Log.h>
#include <plog/Util.h>
#include <plog/Record.h>

namespace oneflow {
  using namespace plog;
  class JsonFmt {
    public:
      static util::nstring header() {
        util::nostringstream ss;

        ss << "["
           << "\"" << "message" << "\", "
           << "\"" << "ts"      << "\", "
           << "\"" << "tid"     << "\", "
           << "\"" << "level"   << "\", "
           << "\"" << "file"    << "\", "
           << "\"" << "line"    << "\", "
           << "\"" << "func"    << "\""
           << "]"  << "\n";
        return ss.str();
      }

      static util::nstring format(const Record& record) {
        util::nostringstream ss;

        ss << "{"
           << "\"message\": " << "\"" << record.getMessage() << "\", "
           << "\"ts\": "      << "\"" << record.getTime().time*1000 + record.getTime().millitm << "\", "
           << "\"tid\": "     << "\"" << record.getTid() << "\", "
           << "\"level\": "   << "\"" << severityToString(record.getSeverity()) << "\", "
           << "\"file\": "    << "\"" << record.getFile() << "\", "
           << "\"line\": "    << "\"" << record.getLine() << "\", "
           << "\"func\": "    << "\"" << record.getFunc() << "@" << record.getObject() << "\""
           << "}" << "\n";
        return ss.str();
      }
  };
}
#endif

