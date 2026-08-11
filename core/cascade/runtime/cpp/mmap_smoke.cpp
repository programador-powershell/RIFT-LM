#include "mmap_bundle.hpp"
#include <iostream>

#if defined(_WIN32)
// POSIX-only tool (mmap); on Windows it compiles to an explicit skip.
int main() {
  std::cerr << "cascade_mmap_smoke: POSIX-only (mmap); unsupported on Windows\n";
  return 2;
}
#else
int main(int argc, char** argv) {
  if (argc < 2) { std::cerr << "usage: cascade_mmap_smoke <file.cascade>\n"; return 1; }
  try {
    auto b = cascade::BundleView::open_mmap(argv[1]);
    std::cout << "magic=CSCD version=" << b.version << " stages=" << b.n_stages
              << " size=" << b.length << " checksum=ok\n";
    for (auto& s : b.stages) {
      std::cout << "  stage id=" << s.stage_id << " off=" << s.offset << " size=" << s.size << "\n";
    }
    return 0;
  } catch (const std::exception& e) {
    // validation failure (bad magic/version/bounds/checksum) → non-zero exit
    std::cerr << "error: " << e.what() << "\n";
    return 2;
  }
}
#endif
