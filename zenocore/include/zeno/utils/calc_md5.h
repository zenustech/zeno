//
// Created by zh on 2025/6/27.
//

#ifndef ZENO_CALC_MD5_H
#define ZENO_CALC_MD5_H
#include <string>
#define CRYPTOPP_ENABLE_NAMESPACE_WEAK 1
#include <cryptopp/md5.h>
#include <cryptopp/hex.h>

namespace zeno {
static std::string calculateMD5(const std::vector<char>& input) {
    unsigned char digest[CryptoPP::Weak::MD5::DIGESTSIZE];
    CryptoPP::Weak::MD5().CalculateDigest(digest, (const unsigned char*)input.data(), input.size());
    CryptoPP::HexEncoder encoder;
    std::string output;
    encoder.Attach(new CryptoPP::StringSink(output));
    encoder.Put(digest, sizeof(digest));
    encoder.MessageEnd();
    return output;
}
static std::string calculateMD5(const std::string& input) {
    unsigned char digest[CryptoPP::Weak::MD5::DIGESTSIZE];
    CryptoPP::Weak::MD5().CalculateDigest(digest, (const unsigned char*)input.data(), input.size());
    CryptoPP::HexEncoder encoder;
    std::string output;
    encoder.Attach(new CryptoPP::StringSink(output));
    encoder.Put(digest, sizeof(digest));
    encoder.MessageEnd();
    return output;
}
}
#endif //ZENO_CALC_MD5_H
