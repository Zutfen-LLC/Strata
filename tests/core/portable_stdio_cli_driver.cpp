#include "portable_model_fixture.hpp"
#include <cstdlib>
using namespace model_fixture;
namespace {
std::string quote(const std::string &s) {
    std::string out = "'";
    for (char c : s)
        out += c == '\'' ? "'\\''" : std::string(1, c);
    return out + "'";
}
} // namespace
int main(int argc, char **argv) {
    try {
        check(argc == 4, "python, test and binary paths required");
        SparseModel f;
        const auto cmd = quote(argv[1]) + " " + quote(argv[2]) + " " + quote(argv[3]) + " " + quote(f.paths[0]) + " " +
                         quote((f.root / "manifest").string());
        return std::system(cmd.c_str()) == 0 ? 0 : 1;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
