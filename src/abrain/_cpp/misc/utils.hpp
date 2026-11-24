#ifndef KGD_ESHN_CPP_UTILS_HPP
#define KGD_ESHN_CPP_UTILS_HPP

#include <sstream>

namespace kgd::eshn::utils {

/// Builds a string from the arguments, delegating type conversion to
///  appropriate operator<<
template <typename... ARGS>
std::string mergeToString (ARGS... args) {
    std::ostringstream oss;
    (oss << ... << std::forward<ARGS>(args));
    return oss.str();
}

/// Manages indentation for provided ostream
/// \author James Kanze @ https://stackoverflow.com/a/9600752
class IndentingOStreambuf : public std::streambuf {
  static constexpr unsigned int DEFAULT_INDENT = 2;   ///< Default indenting value

  std::ostream*       _owner;   ///< Associated ostream
  std::streambuf*     _buffer;  ///< Associated buffer
  bool                _isAtStartOfLine; ///< Whether to insert indentation

  const std::string   _indent;  ///< Indentation value

protected:
  /// Overrides std::basic_streambuf::overflow to insert indentation at line start
  int overflow (int ch) override {
    if (_isAtStartOfLine && ch != '\n')
      _buffer->sputn(_indent.data(), _indent.size());
    _isAtStartOfLine = (ch == '\n');
    return _buffer->sputc(ch);
  }

public:
  /// Creates a proxy buffer managing indentation level
  explicit IndentingOStreambuf(std::ostream& dest,
                               unsigned int spaces = DEFAULT_INDENT)
    : _owner(&dest), _buffer(dest.rdbuf()),
      _isAtStartOfLine(true),
      _indent(spaces, ' ' ) { _owner->rdbuf( this );  }

  /// Returns control of the buffer to its owner
  virtual ~IndentingOStreambuf(void) { _owner->rdbuf(_buffer); }
};

} // end of namespace kgd::eshn::utils

#endif // KGD_ESHN_CPP_UTILS_HPP
