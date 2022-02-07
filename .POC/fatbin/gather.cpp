#include <filesystem>
namespace fs = std::filesystem;

std::vector<std::string> getAllPTXFilesUnder(std::string const &dirpath) {
    std::vector<std::string> res;

    for (auto const &entry: fs::directory_iterator(dirpath)) {
        auto path = entry.path();
        if (fs::path(path).extension() == ".ptx") {
            std::ifstream fin(path,
                std::ios::in | std::ios::binary | std::ios::ate);
            if (!fin.is_open()) {
                std::cerr << "\nerror: unable to open "
                    << filename << " for reading!\n";
                abort();
            }

            size_t inputSize = (size_t)fin.tellg();
            char *memBlock = new char[inputSize + 1];

            fin.seekg(0, std::ios::beg);
            fin.read(memBlock, inputSize);
            fin.close();

            memBlock[inputSize] = '\0';
            res.emplace_back(memBlock);
            delete memBlock;
        }
    }
    return res;
}
